from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# x=spending amount,visits
X=np.array([
    [1000,2],
    [1500,3],
    [2000,2],
    [5000,6],
    [6000,7],
    [5500,5],
    [12000,12],
    [15000,14],
    [13000,11],
    [3000,4]
])
#scale feature
scaler=StandardScaler()
X_scaler=scaler.fit_transform(X)

#training model
model=KMeans(n_clusters=3,random_state=42)
model.fit(X_scaler)
customer=model.predict(X_scaler)
centroids=model.cluster_centers_


print("customers for 10  customers:",customer)
print("center(centroids):,centroids")

real_centroid=scaler.inverse_transform(centroids)
print("real centers:",real_centroid)

for cluster_id in range(3):
    print(f"\ncluster {cluster_id}")
    
for i in range(len(X)):
    if customer[i]==cluster_id:
        print(f"customer {i+1}- spends {[1][0]}, freq {X[i][1]}")
        


inertia=[]
k_range= range(1,6)
for k in k_range:
    model=KMeans(n_clusters=k,random_state=42,n_init=10)
    model.fit(X_scaler)
    inertia.append(model.inertia_)  #<--KMeans dot inertia
    
plt.figure()
plt.plot(k_range,inertia,marker='o')
plt.title('estalish best k')
plt.xlabel('k= number of cluster')
plt.ylabel('inertia')
plt.savefig("elbow.png")
plt.show()


final_model=KMeans(n_clusters=2,random_state=42,n_init=10)
final_labels=final_model.fit_predict(X_scaler)

real_center=scaler.inverse_transform(final_model.cluster_centers_)

plt.figure(figsize=(7,5))
plt.scatter(X[:,0],X[:,1],c=final_labels,cmap="viridis",label="customers")
plt.scatter(real_center[:,0],real_center[:,1],color="red",marker="X",s=200,label="centroids")
#label each point
for i ,(amount,freq) in enumerate(X):
    plt.text(amount+1,freq+0.1,f"c{final_labels[i]}")


plt.xlabel('spending')
plt.ylabel('shopping frequency')
plt.title('customer segmentation')
plt.legend()
plt.grid()
plt.savefig("cluster.png")
plt.show()



print(f"centroids 0:spends ${real_center[0,0]:.1f},frequency{real_center[0,1]:.1f}")

print(f"centroids 1: spends ${real_center[1,0]:.1f},frequency{real_center[1,1]}:1f")