import os
import pg8000
import math
import time
import random 
import itertools 
import argparse
import boto3
import pmdarima as pm
import pandas as pd 
import numpy as np 
import warnings
import statsmodels as sm
from dateutil.parser import parse
from statsmodels.tsa.statespace.sarimax import SARIMAX
warnings.filterwarnings('ignore')
import multiprocessing as mp


def get_canada_data():
  
    """
    This function generates Canada data and is extracted from the perfectmile and d_outbound_shipment_package tables.
    It includes the sum of the packages shipped at the weekly level per FSA. 
    """

    print('Getting permissions to work on MOP Redshift data...')
    sts_client = boto3.client('sts', region_name='us-east-1', endpoint_url='https://sts.us-east-1.amazonaws.com')
    assumed_role_object=sts_client.assume_role(RoleArn="arn:aws:iam::587887714624:role/GeoForecastDBAccessRole",RoleSessionName="AssumeRoleSession2")
    credentials=assumed_role_object['Credentials']
    
    print('Connecting to Redshift cluster...')
    rs_resource=boto3.client(
    'redshift',
    region_name='us-east-1',
    aws_access_key_id=credentials['AccessKeyId'],
    aws_secret_access_key=credentials['SecretAccessKey'],
    aws_session_token=credentials['SessionToken'],
    )

    dbname = 'mopdbrs2'
    dbuser = 'geo_forecast'
    clust_id = 'mopdbrs2'
    rs_host = 'mopdbrs2.cgme0p7hprbm.us-east-1.redshift.amazonaws.com'
    rs_port = 8192
    pwresp = rs_resource.get_cluster_credentials(DbUser=dbuser,DbName=dbname,ClusterIdentifier=clust_id)
    conn = pg8000.connect(database=dbname, host=rs_host, port=rs_port, user=pwresp['DbUser'], password=pwresp['DbPassword'], ssl=True)
    
    sql = """
    WITH pka_dosp AS (SELECT trunc(dateadd('day', -1, date_trunc('week', dateadd('day', 1, ship_day)))) AS ship_week
        ,SUBSTRING(shipping_address_postal_code,1,3) AS zipcode
        ,OTM_OBCUST_PKG_REC_ID
        ,warehouse_id
        ,CASE 
          WHEN customer_ship_option like '%%vendor%%' THEN 1000  /* vendor return packages: only captured for calculating UPS total packages */ 
          WHEN marketplace_id in (1,157860,926620) then 1  /* AFN packages */
          WHEN marketplace_id in (1034080,188630,1065810,1119740,190640) then 100  /* Zappos packages */
          WHEN marketplace_id IN (7) THEN 7 /*CA packages*/
          ELSE 999 /* Other Affiliates */
        END AS marketplace_id
        FROM booker.d_outbound_shipment_packages_na
        WHERE region_id = 1
          AND legal_entity_id = 115
          AND customer_ship_option is not null
          AND upper(ship_method) NOT IN ('UNKNOWN', 'MERCHANT', 'CLEANUP', 'LIQUIDATION_VENDOR_PICKUP','INBOUND_FBA_PCP_FEDEX_GROUND','MAGAZINE_SUBSCRIPTION','PICKUP','-1')
          AND upper(warehouse_id) NOT IN ('PTOP')
          AND shipping_address_country_code = 'CA'
          AND ship_week < '2021-05-02'
          AND ship_week >= trunc(dateadd('day', -1, date_trunc('week',dateadd('day', 1, dateadd('week',-212, GETDATE())))))
          AND marketplace_id=7
        UNION
         SELECT trunc(dateadd('day', -1, date_trunc('week', dateadd('day', 1, ship_date)))) AS ship_week
        ,SUBSTRING(destination_postal_code,1,3) AS zipcode
        ,fulfillment_shipment_id AS OTM_OBCUST_PKG_REC_ID
        ,originating_fulfillment_center AS warehouse_id
        ,CASE 
          WHEN customer_ship_option like '%%vendor%%' THEN 1000  /* vendor return packages: only captured for calculating UPS total packages */ 
          WHEN marketplace_id in (1,157860,926620) then 1  /* AFN packages */
          WHEN marketplace_id in (1034080,188630,1065810,1119740,190640) then 100  /* Zappos packages */
          WHEN marketplace_id IN (7) THEN 7 /*CA packages*/
          ELSE 999 /* Other Affiliates */
        END AS marketplace_id
        FROM perfectmile_ext.d_perfectmile_pkg_attributes_v2_na
        WHERE region_id = 1
          AND legal_entity_id = 115
          AND customer_ship_option is not null
          AND upper(ship_method) NOT IN ('UNKNOWN', 'MERCHANT', 'CLEANUP', 'LIQUIDATION_VENDOR_PICKUP','INBOUND_FBA_PCP_FEDEX_GROUND','MAGAZINE_SUBSCRIPTION','PICKUP','-1')
          AND upper(originating_fulfillment_center) NOT IN ('PTOP')
          AND destination_country_code = 'CA'
          AND ship_week < trunc(dateadd('day', -1, date_trunc('week', dateadd('day', 1, GETDATE()))))
          AND ship_week >= trunc(dateadd('day', -1, date_trunc('week',dateadd('day', 1, dateadd('week',-212, GETDATE())))))
          AND ship_week >= '2021-05-02'
          AND marketplace_id=7
        )


    SELECT ship_week
        ,pka_dosp.zipcode
        ,count(OTM_OBCUST_PKG_REC_ID) as pkgs
    FROM pka_dosp
        INNER JOIN trans_dims_ddl.warehouses w
        ON pka_dosp.warehouse_id = w.warehouse_id
        AND w.legal_entity_id = 115
        AND w.is_sortcenter = 'N'                                           
        AND w.is_prime_now = 'N'                                            
        AND w.is_fresh = 'N'
        and zipcode in (select "left"(destination_postal_code,3) as fsa
    from perfectmile.d_perfectmile_pkg_attributes_v2_na
    where marketplace_id = 7
    and ship_date >= to_date('20200101','YYYYMMDD')
    and not(left(fsa,1) ~ '^[0-9]')
    group by 1
    having count(*) > 1000)
    GROUP BY 1,
             2
    ORDER BY 1,2;    
        """

    cursor = conn.cursor()
    mod_df=pd.read_sql(sql, con=conn, parse_dates=['ship_week'], index_col='ship_week') 
    return mod_df 


def sarima_model(df):

    """
    This function creates (1,1,1)(0,1,0,52) SARIMA Model, and return NaN if fails 
    input from the model. There are 3years of training data, 1 year of test data, and
    provides 5 year-out demand forecasts.
    """

    zipcode = df[0]
    df = df[1]

    try:        
        train_data = df[-4*212:160]
        test_data = df[-52:]
        fcst_data = df[-3*52:]
        
        train_data1 = train_data.iloc[:,1]
        test_data1 = test_data.iloc[:,1]
        fcst = fcst_data.iloc[:,1]
    
        best_model = SARIMAX(train_data1, order=(1, 1, 1), seasonal_order=(0, 1, 0, 52)).fit(dis=-1)
    
        pred = best_model.predict(start = len(train_data1), end = len(train_data1) + 52 -1)
        pred = pd.DataFrame(pred)
        pred.columns= ['pred_V2']
        pred['zipcode'] = df['zipcode'].unique()[0]
    
        best_model_fct = SARIMAX(fcst, order=(1, 1, 1), seasonal_order=(0, 1, 0, 52)).fit(dis=-1)
    
        fct = best_model_fct.predict( start = len(fcst), end = len(fcst) + 260 -1)
        fct = pd.DataFrame(fct) 
        fct.columns = ['pred_V2']
        fct['zipcode'] = df['zipcode'].unique()[0]
    
    except:
        pred = np.full(len(test_data1), np.nan)
        pred = pd.DataFrame(pred)
        pred.columns= ['pred_V2']
        pred['zipcode'] = df['zipcode'].unique()[0]
        fct = np.full(len(fcst), np.nan)
        fct = pd.DataFrame(fct) 
        fct.columns = ['pred_V2']
        fct['zipcode'] = df['zipcode'].unique()[0]
    
    return pred, fct



def post_processing(results):

    """
    This function processed the results, append the results in a list (results_fct), and 
    stores it a dataframe (fct_rst). The output will be in fraction and then it's converted 
    to a csv file (canada_fct.csv)
    """

    results_pred=[]
    results_fct=[]
    for res in results:
        pred = res[0]
        fct = res[1]
        results_pred.append(pred)
        results_fct.append(fct)
        
    fct_rst = pd.DataFrame(columns=['index', 'pred_V2', 'zipcode'])
    fct_rst
    for i in range(len(results_fct)):
        fct_rst = fct_rst.append([results_fct[i].reset_index()])    

    fct_rst['index'] = pd.to_datetime(fct_rst['index'], errors='coerce')

    fct_rst.rename({'index': 'ship_week'}, axis=1, inplace=True)
    fct_rst.rename({'pred_V2': 'forecast'}, axis=1, inplace=True)      

    fct_rst = fct_rst[['ship_week', 'zipcode', 'forecast']]
    fct_rst.loc[fct_rst.forecast<0,  'forecast'] = 0
    pct = fct_rst.groupby(['ship_week','zipcode']).sum()/fct_rst.groupby(['ship_week']).sum()      
    canada_pct = pct.reset_index()
    canada_pct.to_csv('canada_fct.csv', index=False)
    return canada_pct


def main():

    # Read in command line arguments
    parser = argparse.ArgumentParser(description='CanadaForecastingModel Command Line')
    parser.add_argument('--inputs', '-i', default='inputs/')
    parser.add_argument('--outputs', '-o', default='outputs/')
    args = parser.parse_args()

    # Define output file path
    forecast_file = os.path.join(args.outputs, 'canada_forecast.csv')

    print('Generate Canada Data...')
    mod_df = get_canada_data()
    p = mp.Pool(mp.cpu_count())
    print('Generate Forecast Results...')
    results = p.map(sarima_model, mod_df.groupby('zipcode'))
    print('Generate Canada Forecast CSV File...')
    canada_pct=post_processing(results) 
    canada_pct.to_csv('forecast_file')

    return

if __name__ == '__main__':
    main()








