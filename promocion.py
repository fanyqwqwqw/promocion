from flask import Flask, request, jsonify  # Importación de Flask para crear la API REST
import json
import requests
from sqlalchemy import create_engine, text  # Conexión a la base de datos con SQLAlchemy
import pandas as pd  # Manejo de datos con Pandas
import numpy as np  # Operaciones numéricas con NumPy
from sklearn.preprocessing import MinMaxScaler  # Normalización de datos
from sklearn.ensemble import RandomForestRegressor  # Modelo de Machine Learning
from openai import OpenAI
import os


from flask_cors import CORS
import platform


app = Flask(__name__)  # Inicialización de la aplicación Flask
CORS(app) 
# Configuración de conexión a SQL Server
DATABASE_CONFIG = {
    'driver': 'ODBC Driver 17 for SQL Server' if platform.system() == 'Windows' else 'FreeTDS',
    'server': 'riccos_db.mssql.somee.com',
    'database': 'riccos_db',
    'username': 'fanny1010_SQLLogin_1',
    'password': 'aajsv5zdwp',
}


# Cadena de conexión a la base de datos
""" CONNECTION_STRING = (
    f"mssql+pyodbc://{DATABASE_CONFIG['username']}:{DATABASE_CONFIG['password']}@"
    f"{DATABASE_CONFIG['server']}/{DATABASE_CONFIG['database']}?driver={DATABASE_CONFIG['driver']}"
) """
CONNECTION_STRING = (
    f"mssql+pymssql://{DATABASE_CONFIG['username']}:{DATABASE_CONFIG['password']}@"
    f"{DATABASE_CONFIG['server']}/{DATABASE_CONFIG['database']}"
)


engine = create_engine(CONNECTION_STRING)  # Creación del motor de conexión con SQLAlchemy


def obtener_productos():
    """Obtiene productos desde la base de datos considerando historial de pedidos conjuntos."""
    query = text("""
        SELECT p.Id, p.Nombre, p.IdCategoria, p.Precio, p.Stock, p.MargenGanancia,
               ISNULL(MAX(v.FechaVenta), '2000-01-01') AS UltimaVenta,  -- Última fecha de venta
               ISNULL(dp.PorcentajeDescuentoPorUnidad, 0) AS DescuentoPrevio,  -- Descuento aplicado previamente
               COUNT(DISTINCT dp2.IdPedido) AS HistorialPedidosJuntos  -- Número de pedidos donde el producto fue comprado con otros
        FROM producto.Producto p
        JOIN producto.Categoria c ON p.IdCategoria = c.Id
        LEFT JOIN venta.Venta v ON v.IdPedido IN (SELECT IdPedido FROM pedido.DetallePedido WHERE IdProducto = p.Id)
        LEFT JOIN pedido.DetallePedido dp ON dp.IdProducto = p.Id
        LEFT JOIN pedido.DetallePedido dp2 ON dp2.IdProducto = p.Id
        WHERE p.Estado = 1 AND p.Disponibilidad = 1 AND c.Estado = 1 AND c.Disponibilidad = 1
        GROUP BY p.Id, p.Nombre, p.IdCategoria, p.Precio, p.Stock, p.MargenGanancia, dp.PorcentajeDescuentoPorUnidad
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)  # Carga los datos en un DataFrame de Pandas
    return df


def entrenar_modelo(df):
    """Entrena un modelo RandomForest para predecir la rotación de productos."""
    df['UltimaVenta'] = pd.to_datetime(df['UltimaVenta'])  # Convierte la fecha a formato datetime
    df['DiasSinVenta'] = (pd.Timestamp.today() - df['UltimaVenta']).dt.days  # Días desde la última venta
    
    X = df[['Stock', 'MargenGanancia', 'DiasSinVenta', 'HistorialPedidosJuntos']]  # Variables predictoras
    y = df['DiasSinVenta']  # Variable objetivo (rotación)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)  # Modelo de regresión con 100 árboles
    model.fit(X, y)  # Entrenamiento del modelo
    df['PrediccionRotacion'] = model.predict(X)  # Predicción de rotación para cada producto
    return df


def calcular_descuentos(df, n):
    """Calcula descuentos asegurando márgenes saludables considerando rotación, descuentos previos y combinaciones frecuentes."""
    df = entrenar_modelo(df)  # Entrena el modelo antes de calcular los descuentos
    
    # Normalización de variables relevantes
    df['RangoStock'] = MinMaxScaler().fit_transform(df[['Stock']])
    df['MargenNormalizado'] = MinMaxScaler().fit_transform(df[['MargenGanancia']])
    df['Rotacion'] = MinMaxScaler().fit_transform(df[['PrediccionRotacion']])
    df['HistorialNormalizado'] = MinMaxScaler().fit_transform(df[['HistorialPedidosJuntos']])
    
    # Cálculo de puntaje para priorizar productos a descontar
    df['Puntaje'] = df['RangoStock'] * 0.2 + df['MargenNormalizado'] * 0.3 + df['Rotacion'] * 0.3 + df['HistorialNormalizado'] * 0.2
    df = df.sort_values(by='Puntaje', ascending=False)  # Ordena por puntaje (productos más relevantes primero)
    
    categorias_seleccionadas = set()  # Controla que no se repitan categorías
    seleccionados = []  # Lista de productos seleccionados para descuento
    
    for _, row in df.iterrows():
        # Asegura que el producto no tenga descuento previo y que no se repitan categorías
        if row['IdCategoria'] not in categorias_seleccionadas and len(seleccionados) < n and row['DescuentoPrevio'] == 0:
            descuento = min(row['MargenGanancia'] * 0.5, 20)  # Calcula el descuento (máximo 20%)
            seleccionados.append({
                'idProducto': row['Id'],
                'nombreProducto': row['Nombre'],
                'precioUnitario': row['Precio'],
                'precioXCant': row['Precio'],
                'cantidad': 1,
                'idCategoria': row['IdCategoria'],
                'descuento': round(descuento, 2)
            })
            categorias_seleccionadas.add(row['IdCategoria'])  # Agrega la categoría a la lista de seleccionadas
        
        if len(seleccionados) >= n:
            break  # Detiene la selección si ya se alcanzó la cantidad solicitada
    
    return seleccionados

# Función para generar descripción con IA
def generar_descripcion(productos):
    return 'descripcion generada por IA openAI que está comentado para no consumir limites'


# Función para generar imagen con IA
def generar_imagen(descripcion):
    return 'url de imagen generada por IA'




@app.route('/lista-productos', methods=['GET'])
def obtener_promociones():
    """Endpoint que devuelve una lista de productos con descuentos sugeridos."""
    try:
        n = int(request.args.get('cantidad', 5))  # Obtiene el número de productos a recomendar (default 5)
        df = obtener_productos()  # Obtiene los productos de la base de datos
        productos_seleccionados = calcular_descuentos(df, n)  # Calcula los descuentos
        return jsonify({'productosSeleccionados': productos_seleccionados})  # Devuelve los productos en formato JSON
    except Exception as e:
        return jsonify({'error': str(e)})  # Manejo de errores



@app.route('/obtener_descripcion', methods=['GET'])
def obtener_descripcion():
    try:
        productos_json = request.args.get('productos')
        if not productos_json:
            return jsonify({"error": "No se proporcionó la lista de productos"}), 400

        productos = json.loads(productos_json)

        # Generar descripción con IA
        descripcion = generar_descripcion(productos)

        # Generar imagen con IA
        url_imagen = generar_imagen(descripcion)

        return jsonify({
            "mensaje": "Promoción generada correctamente",
            "descripcion": descripcion,
            "imagen": url_imagen
        })

    except json.JSONDecodeError:
        return jsonify({"error": "Formato JSON inválido"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
