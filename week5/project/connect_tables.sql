----------------------------------region table------------------------------------------

\c northwind
CREATE TABLE IF NOT EXISTS regions(
region_id SMALLINT,
region_description VARCHAR(20)
);
\copy regions FROM '/home/esharifi/Documents/spiced/ginger-pipeline-student-code/week5/project/northwind_data_clean/data/regions.csv' DELIMITER ',' CSV HEADER;

-------------------------------------- territories table --------

CREATE TABLE IF NOT EXISTS territories(
territory_id INT,
territoriy_description VARCHAR(20),
region_id SMALLINT
);
\copy territories FROM '/home/esharifi/Documents/spiced/ginger-pipeline-student-code/week5/project/northwind_data_clean/data/territories.csv' DELIMITER ',' CSV HEADER;

-------------------------------------- categories table --------

--pictures are binary strings and in postgres bytea is the right type for supporting binary strings

CREATE TABLE IF NOT EXISTS categories(
category_id SMALLINT,
category_name VARCHAR(40),
description TEXT,
picture BYTEA
);
\copy categories FROM '/home/esharifi/Documents/spiced/ginger-pipeline-student-code/week5/project/northwind_data_clean/data/categories.csv' DELIMITER ',' CSV HEADER;

-------------------------------------- customers table --------

CREATE TABLE IF NOT EXISTS customers(
customer_id VARCHAR(30),
company_name Text,
contact_name VARCHAR(50),
contact_title VARCHAR(50),
address TEXT,
city VARCHAR(30),
region VARCHAR(30),
postal_code VARCHAR(20),
country VARCHAR(30),
phone VARCHAR(40),
fax VARCHAR(40)
);
\copy customers FROM '/home/esharifi/Documents/spiced/ginger-pipeline-student-code/week5/project/northwind_data_clean/data/customers.csv' DELIMITER ',' CSV HEADER NULL AS 'NULL';

--------------------------------------********** employees table *************--------

CREATE TABLE IF NOT EXISTS employees(
employee_id SMALLINT,
last_name VARCHAR(30),
first_name VARCHAR(30),
title TEXT,
title_of_courtesy VARCHAR(10),
birth_date TIMESTAMP,
hire_date TIMESTAMP,
address TEXT,
city VARCHAR(30),
region VARCHAR(30),
postal_code VARCHAR(40),
country VARCHAR(20),
home_phone VARCHAR(40),
extension SMALLINT,
photo BYTEA,
notes TEXT,
reports_to SMALLINT,
photo_path TEXT
);
\copy employees(employee_id, last_name, first_name, title ,title_of_courtesy, birth_date, hire_date, address, city, region, postal_code, country, home_phone, extension, photo, notes, reports_to, photo_path) FROM '/home/esharifi/Documents/spiced/ginger-pipeline-student-code/week5/project/northwind_data_clean/data/employees.csv' DELIMITER ',' CSV HEADER NULL AS 'NULL';

-------------------------------------- employee_territories table --------

CREATE TABLE IF NOT EXISTS employee_territories(
employee_id SMALLINT,
territory_id INT
);
\copy employee_territories FROM '/home/esharifi/Documents/spiced/ginger-pipeline-student-code/week5/project/northwind_data_clean/data/employee_territories.csv' DELIMITER ',' CSV HEADER;

-------------------------------------- order_details table --------

CREATE TABLE IF NOT EXISTS order_details(
order_id INT,
product_id INT,
unit_price FLOAT,
quantity SMALLINT,
discount float
);
\copy order_details FROM '/home/esharifi/Documents/spiced/ginger-pipeline-student-code/week5/project/northwind_data_clean/data/order_details.csv' DELIMITER ',' CSV HEADER;


-------------------------------------- orders table --------

CREATE TABLE IF NOT EXISTS orders(
order_id INT,
customer_id VARCHAR(20),
employee_id SMALLINT,
order_date TIMESTAMP,
required_date TIMESTAMP,
shipped_date TIMESTAMP,
ship_via SMALLINT,
freight FLOAT,
ship_name VARCHAR(100),
ship_address TEXT,
ship_city VARCHAR(40),
ship_region VARCHAR(30),
ship_postal_code VARCHAR(50),
ship_country VARCHAR(50)
);
\copy orders FROM '/home/esharifi/Documents/spiced/ginger-pipeline-student-code/week5/project/northwind_data_clean/data/orders.csv' DELIMITER ',' CSV HEADER NULL AS 'NULL';

-------------------------------------- products table --------
DROP TABLE products;
CREATE TABLE IF NOT EXISTS products(
product_id SMALLINT,
product_name VARCHAR(100),
supplier_id SMALLINT,
category_id SMALLINT,
quantity_per_unit TEXT,
unit_price FLOAT,
units_in_stock SMALLINT,
units_on_order SMALLINT,
reorder_level SMALLINT,
discontinued SMALLINT
);
\copy products FROM '/home/esharifi/Documents/spiced/ginger-pipeline-student-code/week5/project/northwind_data_clean/data/products.csv' DELIMITER ',' CSV HEADER NULL AS 'NULL';


-------------------------------------- shippers table --------

CREATE TABLE IF NOT EXISTS shippers(
shipper_id SMALLINT,
company_name VARCHAR(40),
phone VARCHAR(20)
);
\copy shippers FROM '/home/esharifi/Documents/spiced/ginger-pipeline-student-code/week5/project/northwind_data_clean/data/shippers.csv' DELIMITER ',' CSV HEADER;

-------------------------------------- suppliers table --------

CREATE TABLE IF NOT EXISTS suppliers(
supplier_id SMALLINT,
company_name VARCHAR(70),
contact_name VARCHAR(40),
contact_title VARCHAR(70),
address TEXT,
city VARCHAR(40),
region VARCHAR(40),
postal_code VARCHAR(20),
country VARCHAR(40),
phone VARCHAR(30),
fax VARCHAR(30),
home_page TEXT
);
\copy suppliers FROM '/home/esharifi/Documents/spiced/ginger-pipeline-student-code/week5/project/northwind_data_clean/data/suppliers.csv' DELIMITER ',' CSV HEADER;


