# One-vs-all SVM
Метод опорных векторов, добавляю квадратичные признаки, скалирование - деление всего на 255 (до добавления новых признаков). Использую стохастический градиентный спуск, градиент считается по одному объекту.
Запуск как прописано в условии задания.
Результаты на 100 эпохах (обучение ~20 минут):
### Train: 
                  precision    recall  f1-score   support

               0       0.95      0.98      0.97      5923
               1       0.96      0.99      0.97      6742
               2       0.94      0.90      0.92      5958
               3       0.91      0.92      0.92      6131
               4       0.94      0.94      0.94      5842
               5       0.93      0.89      0.91      5421
               6       0.95      0.97      0.96      5918
               7       0.96      0.94      0.95      6265
               8       0.91      0.92      0.92      5851
               9       0.90      0.92      0.91      5949

       micro avg       0.94      0.94      0.94     60000
       macro avg       0.94      0.94      0.94     60000
    weighted avg       0.94      0.94      0.94     60000


    

### Test: 
                  precision    recall  f1-score   support

               0       0.95      0.98      0.97       980
               1       0.95      0.99      0.97      1135
               2       0.95      0.88      0.91      1032
               3       0.89      0.92      0.90      1010
               4       0.92      0.93      0.93       982
               5       0.93      0.87      0.90       892
               6       0.94      0.95      0.94       958
               7       0.95      0.92      0.94      1028
               8       0.89      0.92      0.90       974
               9       0.91      0.92      0.92      1009

       micro avg       0.93      0.93      0.93     10000
       macro avg       0.93      0.93      0.93     10000
    weighted avg       0.93      0.93      0.93     10000

Если поднять число эпох, то результаты улучшатся.
