import mysql.connector
from mysql.connector import Error

class DatabaseConnector:
    def __init__(self, host, user, password, database):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = None

    def connect(self):
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            print("Successfully connected to the database")
        except Error as e:
            print(f"Error connecting to the database: {e}")

    def disconnect(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("Database connection closed")

    def execute_query(self, query, params=None):
        try:
            if not self.connection or not self.connection.is_connected():
                self.connect()

            cursor = self.connection.cursor(dictionary=True)
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            results = cursor.fetchall()
            cursor.close()
            return results
        except Error as e:
            print(f"Error executing query: {e}")
            return None

    def fetch_product_data(self, product_ids):
        placeholders = ', '.join(['%s'] * len(product_ids))
        query = f"""
        SELECT id, uniq_id, product_name, product_category_tree, retail_price, 
               discounted_price, image, is_FK_Advantage_product, description, 
               product_rating, overall_rating, brand, product_specifications
        FROM products
        WHERE id IN ({placeholders})
        """
        results = self.execute_query(query, product_ids)
        
        if results:
            return {str(row['id']): row for row in results}
        return {}

# # Usage example
# if __name__ == "__main__":
#     # Replace these with your actual database credentials
#     db_config = {
#         "host": "172.16.0.10",
#         "user": "project",
#         "password": "iiitg@abc",
#         "database": "ecommerce"
#     }

#     db = DatabaseConnector(**db_config)
#     db.connect()

#     # Example usage with Pinecone IDs
#     pinecone_ids = ["1", "2", "3", "4", "5"]  # Replace with actual IDs from Pinecone
#     product_data = None
#     product_data = db.fetch_product_data([int(id) for id in pinecone_ids])

#     if product_data:
#         for id, data in product_data.items():
#             print(f"Product ID: {id}")
#             for key, value in data.items():
#                 print(f"{key}: {value}")
#             print("---")

#     db.disconnect()