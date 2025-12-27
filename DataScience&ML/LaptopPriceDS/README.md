# Laptop Price Analysis & Data Science
Laptop is an important part of my daily life. Therefore, this gave me the idea and motivation to do this experiment.

This project requires transforming the decision-making process into a model.. For example, e-commerce websites often offer sorting by price, brand, etc. This is a very simple decision-making method because it indicates whether you value price or brand. More advanced methods can also allow consumers to choose usage scenarios, bundled offers, etc. It's essentially a series of filters. My decision-making method is also simple: once the model is trained and converted into a model that can determine the price of mobile phones, then sort by price, and finally, I adjust the settings according to my needs.

Data science problem: Build a regression model for pricing and select laptops with high cost-performance ratios based on the pricing model.



## Data Frame

1.  Company- String -Laptop Manufacturer  
2.  Product -String -Brand and Model  
3. TypeName -String -Type (Notebook, Ultrabook, Gaming, etc.) 
4. Inches -Numeric- Screen Size 
5. ScreenResolution -String- Screen Resolution  
6. Cpu- String -Central Processing Unit (CPU) 
7. Ram -String- Laptop RAM 
8. Memory -String- Hard Disk / SSD Memory 
9. GPU -String- Graphics Processing Units (GPU) 
10. OpSys -String- Operating System 
11. Weight -String- Laptop Weight 
12. Price_euros -Numeric- Price (Euro)  


|   laptop_ID | Company   | Product     | TypeName   |   Inches | ScreenResolution                   | Cpu                  | Ram   | Memory              | Gpu                          | OpSys   | Weight   |   Price_euros |
|------------:|:----------|:------------|:-----------|---------:|:-----------------------------------|:---------------------|:------|:--------------------|:-----------------------------|:--------|:---------|--------------:|
|           1 | Apple     | MacBook Pro | Ultrabook  |     13.3 | IPS Panel Retina Display 2560x1600 | Intel Core i5 2.3GHz | 8GB   | 128GB SSD           | Intel Iris Plus Graphics 640 | macOS   | 1.37kg   |       1339.69 |
|           2 | Apple     | Macbook Air | Ultrabook  |     13.3 | 1440x900                           | Intel Core i5 1.8GHz | 8GB   | 128GB Flash Storage | Intel HD Graphics 6000       | macOS   | 1.34kg   |        898.94 |
## Data Analysis
Columns : 13 , Rows :  1303

| Missing values   |   0 |
|:-----------------|----:|
| laptop_ID        |   0 |
| Company          |   0 |
| Product          |   0 |
| TypeName         |   0 |
| Inches           |   0 |
| ScreenResolution |   0 |
| Cpu              |   0 |
| Ram              |   0 |
| Memory           |   0 |
| Gpu              |   0 |
| OpSys            |   0 |
| Weight           |   0 |
| Price_euros      |   0 |

Make sure there's no missing value or type error.  
And also no any data repeat.
<br> <br> 


|       |   laptop_ID |    Inches |   Price_euros |
|:------|------------:|----------:|--------------:|
| count |    1303     | 1303      |      1303     |
| mean  |     660.156 |   15.0172 |      1123.69  |
| std   |     381.172 |    1.4263 |       699.009 |
| min   |       1     |   10.1    |       174     |
| 25%   |     331.5   |   14      |       599     |
| 50%   |     659     |   15.6    |       977     |
| 75%   |     990.5   |   15.6    |      1487.88  |
| max   |    1320     |   18.4    |      6099     |

---

### Basic Statistical Summary (EDA)
|       |   laptop_ID |    Inches |   Price_euros |
|:------|------------:|----------:|--------------:|
| count |    1303     | 1303      |      1303     |
| mean  |     660.156 |   15.0172 |      1123.69  |
| std   |     381.172 |    1.4263 |       699.009 |
| min   |       1     |   10.1    |       174     |
| 25%   |     331.5   |   14      |       599     |
| 50%   |     659     |   15.6    |       977     |
| 75%   |     990.5   |   15.6    |      1487.88  |
| max   |    1320     |   18.4    |      6099     |
---
|        | Company   | Product   | TypeName   | ScreenResolution   | Cpu                        | Ram   | Memory    | Gpu                   | OpSys      | Weight   |
|:-------|:----------|:----------|:-----------|:-------------------|:---------------------------|:------|:----------|:----------------------|:-----------|:---------|
| count  | 1303      | 1303      | 1303       | 1303               | 1303                       | 1303  | 1303      | 1303                  | 1303       | 1303     |
| unique | 19        | 618       | 6          | 40                 | 118                        | 9     | 39        | 110                   | 9          | 179      |
| top    | Dell      | XPS 13    | Notebook   | Full HD 1920x1080  | Intel Core i5 7200U 2.5GHz | 8GB   | 256GB SSD | Intel HD Graphics 620 | Windows 10 | 2.2kg    |
| freq   | 297       | 30        | 727        | 507                | 190                        | 619   | 412       | 281                   | 1072       | 121      |


Locate to [EDA Visualization](EDA/)


### Feature Engineering


### Machine Learning


