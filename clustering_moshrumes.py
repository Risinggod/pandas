import pandas as pd
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.preprocessing import StandardScaler

path = "/Users/student/PycharmProjects/pandas/mushrooms.csv"

data = pd.read_csv(path, delimiter=",")

#data_unknown = data.drop(['class'], axis=1)
data_unknown = data
data_unknown = data_unknown.astype("category")
data_unknown = data_unknown.apply(lambda x: x.cat.codes)

s_scaler = StandardScaler()

#data_unknown = pd.DataFrame(s_scaler.fit_transform(data_unknown), columns=data_unknown.columns)

print(data_unknown)


model = KMeans()

visualizer = KElbowVisualizer(model, K=(2, 9), timings=False)
visualizer.fit(data_unknown)
visualizer.show()

KMeans = KMeans(n_clusters=6)

pred = KMeans.fit_predict(data_unknown)

data_new = pd.concat([data, pd.DataFrame(pred, columns=["label"])], axis=1)
print(data_new)

data_new.to_csv("./data_new_moshrumes.csv")
