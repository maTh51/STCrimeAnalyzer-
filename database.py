import jaydebeapi
import pandas as pd

class DatabaseConnection:
    
    def __init__(self, credentials_file: str = '.env/.credentials_bisp'):
        with open(credentials_file, 'r') as f:
            credentials = f.read().split('\n')
            self.user = credentials[0]
            self.pwd  = credentials[1]
    
    def execute_query(self, query: str):
        conn = jaydebeapi.connect(
            "com.cloudera.impala.jdbc.Driver",
            "jdbc:impala://clouderacdp02.prodemge.gov.br:21051/db_reds_reporting;AuthMech=3;SSL=1",
            {
                "UID": self.user,
                "PWD": self.pwd,
            },
            "config/ImpalaJDBC41.jar",
        )

        curs = conn.cursor()
        curs.execute(query)

        columns = [col[0] for col in curs.description]
        results = curs.fetchall()
        df = pd.DataFrame(results, columns=columns)

        curs.close()
        conn.close()

        return df

    def construct_query_where(self, filter: dict):
        query = []
        for k,v in filter.items():
            if k == "exclude":
                for ki, vi in v.items():
                    if type(vi) == list:
                        if len(vi) == 1: vi.append(vi[0])
                        query.append(f"{ki} NOT IN {tuple(vi)}")
                    else:
                        query.append(f"{ki} != '{vi}'")
            elif k == "start_date":
                query.append(f"data_hora_fato >= '{v}'")
            elif k == "end_date":
                query.append(f"data_hora_fato <= '{v}'")
            elif k == "crimes":
                if len(v) == 1: v.append(v[0])
                query.append(f"(natureza_descricao IN {tuple(v)} \
                             or natureza_secundaria1_descricao IN {tuple(v)} \
                             or natureza_secundaria2_descricao IN {tuple(v)} \
                             or natureza_secundaria3_descricao IN {tuple(v)})")
            elif type(v) == int:
                query.append(f"{k} = {v}")
            elif type(v) == list:
                if len(v) == 1: v.append(v[0])
                query.append(f"{k} IN {tuple(v)}")
            else:
                query.append(f"{k} = '{v}'")

        return ' and '.join(query)
    

    def get_data_grouped(self, columns: list, filter: dict = {'ocorrencia_uf': 'MG'}):
        query = f" SELECT {', '.join(columns)}, COUNT(*) AS qtd \
        FROM db_bisp_reds_reporting.tb_ocorrencia \
        WHERE {self.construct_query_where(filter)} \
        GROUP BY {', '.join(columns)} \
        "

        return self.execute_query(query)

    def get_data_all(self, column: str, filter: dict = {'ocorrencia_uf': 'MG'}, online=True):
        if online:
            query = f" SELECT {', '.join(column)}\
            FROM db_bisp_reds_reporting.tb_ocorrencia \
            WHERE {self.construct_query_where(filter)} \
            "

            return self.execute_query(query)
        else:
            df = pd.read_csv('/home/yan/IC-CrimeAnalytics-modified/API/experimental/dados_BH.csv')
            return df