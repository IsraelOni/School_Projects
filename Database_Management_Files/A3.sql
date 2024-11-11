CREATE DATABASE IF NOT EXISTS BOOKS;

USE BOOKS;
CREATE TABLE authors (
    author_id INT PRIMARY KEY,
    author_name VARCHAR(100)
);

-- Inserting sample data into authors table
INSERT INTO authors (author_id, author_name) VALUES
(1, 'John Smith'),
(2, 'Emily Johnson'),
(3, 'Michael Williams');

-- Creating the genres table
CREATE TABLE genres (
    genre_id INT PRIMARY KEY,
    genre_name VARCHAR(50)
);

-- Inserting sample data into genres table
INSERT INTO genres (genre_id, genre_name) VALUES
(1, 'Fiction'),
(2, 'Mystery'),
(3, 'Romance');

-- Creating the books table
CREATE TABLE books (
    book_id INT PRIMARY KEY,
    book_title VARCHAR(200),
    author_id INT,
    genre_id INT,
    price DECIMAL(10, 2),
    FOREIGN KEY (author_id) REFERENCES authors(author_id),
    FOREIGN KEY (genre_id) REFERENCES genres(genre_id)
);

-- Inserting sample data into books table
INSERT INTO books (book_id, book_title, author_id, genre_id, price) VALUES
(101, 'The Lost Symbol', 1, 1, 15.99),
(102, 'Gone Girl', 2, 2, 12.50),
(103, 'Pride and Prejudice', 3, 3, 10.99);


-- Creating the customers table
CREATE TABLE customers (
    customer_id INT PRIMARY KEY,
    customer_name VARCHAR(100),
    email VARCHAR(100)
);

-- Inserting sample data into customers table
INSERT INTO customers (customer_id, customer_name, email) VALUES
(1, 'Alice Johnson', 'alice@example.com'),
(2, 'Bob Smith', 'bob@example.com'),
(3, 'Eva Williams', 'eva@example.com');

-- Creating the orders table
CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    customer_id INT,
    book_id INT,
    quantity INT,
    order_date DATE,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    FOREIGN KEY (book_id) REFERENCES books(book_id)
);

-- Inserting sample data into orders table
INSERT INTO orders (order_id, customer_id, book_id, quantity, order_date) VALUES
(1, 1, 101, 2, '2024-03-01'),
(2, 2, 102, 1, '2024-03-02'),
(3, 1, 103, 3, '2024-03-03'),
(4, 3, 101, 1, '2024-03-03');

-- This is my query statements which includes 5 query statements


-- Q1
SELECT c.customer_name, g.genre_name, b.book_title
FROM customers c
JOIN ( SELECT  o.customer_id, b.genre_id
	   FROM orders o, books b
       WHERE o.book_id = b.book_id) As orderedGenre ON c.customer_id = orderedGenre.customer_id
JOIN genres g ON orderedGenre.genre_id = g.genre_id
JOIN books b ON g.genre_id = b.genre_id
WHERE b.book_id NOT IN 
(SELECT o.book_id 
FROM orders o
WHERE o.customer_id = c.customer_id AND 
o.book_id = b.book_id);
                        

-- Q2
SELECT SUM(b.PRICE * o.QUANTITY)
FROM orders o
JOIN books b ON o.book_id = b.book_id;

-- Q3

SELECT customer_name
FROM (SELECT customer_name,COUNT(order_id)
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
GROUP BY customer_name
ORDER BY COUNT(order_id) DESC) AS customer_order_count
LIMIT 1;

-- Q4
SELECT author_name
FROM (SELECT author_name ,SUM(o.quantity) AS total_books_sold
FROM authors a
JOIN books b ON a.author_id = b.author_id
JOIN orders o  ON b.book_id = o.book_id 
GROUP BY author_name
ORDER BY total_books_sold DESC) AS total_books_sold_by_authors
LIMIT 1;

SELECT book_title
FROM books
ORDER BY price DESC
LIMIT 3;
