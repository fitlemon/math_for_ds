### <img src='../static/img/mipt-icon.png' width="70" height="30"> Курс "Математика для Машинного обучения" (Магистратура "Науки о данных" МФТИ - 2023/2025) 
---
 :house: [Вернуться на главный экран](..)
# Конспект по Модулю #7 **     Обучение без учителя. Кластеризация. Снижение размерности данных**  :blue_book:


## Оглавление конспекта
1. [Обучение без учителя](1_unsupervised.ipynb) 
2. [K-means](2_kmeans.ipynb) 
3. [K-means. Реализация](3_kmeans_realisation.ipynb)  
4. [DBSCAN.](4_dbscan.ipynb)
5. [DBSCAN. Реализация](5_dbscan_realisation.ipynb)
6. [Агломеративная кластеризация](6_aglomeration.ipynb)
7. [Агломеративная кластеризация. Реализация](7_aglomeration_realisation.ipynb)
8. [Метод главных компонент](8_pca.ipynb)
9. [Метод главных компонент. Реализация](9_pca_realisation.ipynb)
10. [t-SNE](10_tsne.ipynb)
11. [t-SNE. Реализация](11_tsne_realisation.ipynb)
12. [Итоги Модуля № 7](12_summary.ipynb)
13. [Ноутбук к семинару по модулю №7](13_seminar_1.ipynb)
---

### Дополнительная информация по модулю: :books:
1. [Кластеризация в scikit-learn](https://scikit-learn.ru/clustering/)
2. [Статья «Кластеризация объектов с помощью алгоритма DBSCAN»](http://pzs.dstu.dp.ua/DataMining/cluster/bibl/%D0%9A%D0%9B%D0%90%D0%A1%D0%A2%D0%95%D0%A0%D0%98%D0%97%D0%90%D0%A6%D0%98%D0%AF%20%D0%9E%D0%91%D0%AA%D0%95%D0%9A%D0%A2%D0%9E%D0%92%20%D0%A1%20%D0%9F%D0%9E%D0%9C%D0%9E%D0%A9%D0%AC%D0%AE%20%D0%90%D0%9B%D0%93%D0%9E%D0%A0%D0%98%D0%A2%D0%9C%D0%90%20DBSCAN.pdf)
3. [Учебник по машинному обучению](https://education.yandex.ru/handbook/ml)
4. [scikit-learn: User Guide](https://scikit-learn.org/stable/user_guide.html)
5. [ML_Econom_2021-2022 . Тема семинара: отбор признаков](https://github.com/Murcha1990/ML_Econom_2021-2022/blob/main/%D0%A1%D0%B5%D0%BC%D0%B8%D0%BD%D0%B0%D1%80%D1%8B/%D0%A1%D0%B5%D0%BC%D0%B8%D0%BD%D0%B0%D1%80%207/Seminar7.ipynb)
6. [tslearn.clustering.KernelKMeans](https://tslearn.readthedocs.io/en/latest/gen_modules/clustering/tslearn.clustering.KernelKMeans.html)
7. [t-SNE in python from scratch ](https://github.com/beaupletga/t-SNE)
8. [In Raw Numpy: t-SNE](https://nlml.github.io/in-raw-numpy/in-raw-numpy-t-sne/)
9. [Singular Value Decomposition (SVD) tutorial](https://web.mit.edu/be.400/www/SVD/Singular_Value_Decomposition.htm)
10. [Selfedu. Yotube. #25. Метод главных компонент (Principal Component Analysis) | Машинное обучение](https://www.youtube.com/watch?v=AoBykkvOMDw&list=PLA0M1Bcd0w8zxDIDOTQHsX68MCDOAJDtj&index=26)
11. [Interactive Linear Algebra. Dan Margalit, Joseph Rabinoff. 5.1Eigenvalues and Eigenvectors](https://textbooks.math.gatech.edu/ila/eigenvectors.html)
12. [Relationship between SVD and PCA. How to use SVD to perform PCA?](https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca)
13. [What are left and right singular vectors in SVD?](https://math.stackexchange.com/questions/3982195/what-are-left-and-right-singular-vectors-in-svd)
14. [Are there cases where PCA is more suitable than t-SNE?](https://stats.stackexchange.com/questions/238538/are-there-cases-where-pca-is-more-suitable-than-t-sne)
15. [esokolov/ml-course-hse/Методы понижения размерности](https://github.com/esokolov/ml-course-hse/blob/master/2022-spring/seminars/sem14_pca_tsne.ipynb)
16. [Visualizing K-Means Clustering](https://www.naftaliharris.com/blog/visualizing-k-means-clustering/)
17. [Visualizing DBSCAN Clustering](https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/)