import os
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from datetime import datetime, timedelta
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import math
import requests

# --- 1. LAS LLAVES DEL CORTIJO (Ahora desde la caja fuerte de GitHub) ---
api_key = os.getenv('ALPACA_API_KEY')
secret_key = os.getenv('ALPACA_SECRET_KEY')

cliente_trading = TradingClient(api_key, secret_key, paper=True)
cliente_datos = StockHistoricalDataClient(api_key, secret_key)

# --- 2. CONTROL DE CAJA Y STOCK ---
cuenta = cliente_trading.get_account()
saldo_disponible = float(cuenta.buying_power)

posiciones_abiertas = cliente_trading.get_all_positions()
cartera_actual = [posicion.symbol for posicion in posiciones_abiertas]

print(f"--- ESTADO DE LA CAJA ---")
print(f"Saldo para gastar: {saldo_disponible:.2f} $")
print(f"Acciones en cartera: {cartera_actual}\n")

# --- 3. EL EMBUDO INTELIGENTE: BUSCANDO EL RASTRO DEL DINERO ---
url_wiki = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
cabecera = {'User-Agent': 'Mozilla/5.0'}
respuesta = requests.get(url_wiki, headers=cabecera).text

tabla_sp500 = pd.read_html(respuesta)[0]
todos_los_simbolos = tabla_sp500['Symbol'].tolist()
todos_los_simbolos = [s for s in todos_los_simbolos if '.' not in s and '-' not in s]

print("Escaneando el mercado en busca de volumen anormal...")

parametros_volumen = StockBarsRequest(
    symbol_or_symbols=todos_los_simbolos,
    timeframe=TimeFrame.Day,
    start=datetime.now() - timedelta(days=40),
    end=datetime.now(),
    feed=DataFeed.IEX 
)

velas_mercado = cliente_datos.get_stock_bars(parametros_volumen)
tabla_mercado = velas_mercado.df

lista_calientes = []
for activo in todos_los_simbolos:
    try:
        if activo in tabla_mercado.index:
            datos = tabla_mercado.loc[activo]
            if len(datos) > 20:
                media_vol = datos['volume'][:-1].mean()
                vol_hoy = datos['volume'].iloc[-1]
                if media_vol > 0:
                    ratio = vol_hoy / media_vol
                    lista_calientes.append({'Activo': activo, 'Ratio': ratio})
    except: continue

df_calientes = pd.DataFrame(lista_calientes).sort_values(by='Ratio', ascending=False)
top_20 = df_calientes.head(20)['Activo'].tolist()
print(f"Radar detectó movimiento en: {top_20}")

# --- 4. DECISIONES Y ÓRDENES ---
presupuesto_por_accion = saldo_disponible * 0.15

for activo in top_20:
    try:
        datos_activo = tabla_mercado.loc[activo].copy()
        datos_activo['retorno'] = datos_activo['close'].pct_change()
        datos_activo['objetivo'] = (datos_activo['retorno'].shift(-1) > 0).astype(int)
        datos_activo['media_5d'] = datos_activo['close'].rolling(window=5).mean()
        datos_activo['media_20d'] = datos_activo['close'].rolling(window=20).mean()
        datos_activo = datos_activo.dropna()
        
        X = datos_activo[['media_5d', 'media_20d', 'retorno']]
        y = datos_activo['objetivo']
        modelo = RandomForestClassifier(n_estimators=100, random_state=42)
        modelo.fit(X[:-1], y[:-1])
        
        decision = modelo.predict(X.iloc[-1:])[0]
        precio = datos_activo['close'].iloc[-1]
        
        if decision == 0 and activo in cartera_actual:
            cliente_trading.close_position(activo)
            print(f"[{activo}] Señal QUIETO. Vendiendo.")
        elif decision == 1 and activo not in cartera_actual:
            cant = math.floor(presupuesto_por_accion / precio)
            if cant > 0:
                orden = MarketOrderRequest(symbol=activo, qty=cant, side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
                cliente_trading.submit_order(order_data=orden)
                print(f"[{activo}] Señal COMPRA. Orden enviada por {cant} acciones.")
        else:
            print(f"[{activo}] Sin cambios.")
    except: continue

print("\n--- OPERATIVA TERMINADA ---")
