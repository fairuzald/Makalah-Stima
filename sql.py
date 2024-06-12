import mysql.connector
import time

# Establish connection to your MySQL database
mydb = mysql.connector.connect(
    host="Fairuz",
    user="root",
    password="bismillah.33",
    database="cafe3"
)
mycursor = mydb.cursor()
# Non-optimized query
non_optimized_query = """
SELECT *
FROM Transaksi T
JOIN DetailTransaksi DT ON T.nomorTransaksi = DT.nomorTransaksi
JOIN Menu M ON DT.idMenu = M.idMenu
JOIN RatingMenu RM ON DT.idDetailTransaksi = RM.idDetailTransaksi
WHERE T.tanggalTransaksi BETWEEN '2022-01-01' AND '2022-06-30'
AND M.tipe = 'Makanan'
AND RM.rating >= 4;
"""

# Execute the non-optimized query and measure execution time
start_time = time.time()
mycursor.execute(non_optimized_query)
non_optimized_result = mycursor.fetchall()
end_time = time.time()
non_optimized_execution_time = end_time - start_time

# Divide and Conquer approach
# Sub-query 1: Get transaction numbers within a specific date range
subquery1 = """
SELECT nomorTransaksi
FROM Transaksi
WHERE tanggalTransaksi BETWEEN '2022-01-01' AND '2022-06-30'
"""

# Sub-query 2: Get detail transaction IDs with high ratings
subquery2 = """
SELECT idDetailTransaksi
FROM RatingMenu
WHERE rating >= 4
"""

# Combine sub-queries and execute the optimized query
optimized_query = f"""
SELECT *
FROM DetailTransaksi
WHERE nomorTransaksi IN ({subquery1})
AND idDetailTransaksi IN ({subquery2});
"""

start_time = time.time()
mycursor.execute(optimized_query)
optimized_result = mycursor.fetchall()
end_time = time.time()
optimized_execution_time = end_time - start_time

# Compare the execution times
print("Non-optimized execution time:", non_optimized_execution_time, "seconds", len(non_optimized_result))
print("Optimized execution time:", optimized_execution_time, "seconds", len(optimized_result))

# Close the database connection
mycursor.close()
mydb.close()