Toroidal k-means by Simon Wilshin, Jan 2016

So I frequently work with variables where 0 and 100% are the same thing (if 
you are 100% of the way through a cycle that is identical to being 0% through 
the next cycle). These variables are on a torus, and I sometimes need to 
cluster them. Unfortunately I couldn't find any software that took into 
account this topological tomfoolery, so I wrote my own.

I've called it 'Toroidal k-means, but really it is toroidal k-'some suitable 
measure of central tendency' as you can also use it for k-medians, which I 
sometimes find works better (like in the usage example I provide). With this 
bit of software you can:

 - Cluster 2D data on a torus with variable metric
 - Use a variety of measures of central tendency including k-means and k-medians

Code is documented and a usage example is included if the toroidalkmeans.py 
is ran as main. It generates two clusters in two dimensions, and clusters 
them with scipy's kmeans2 function, and this toroidal k-means and k-medians 
method. This is very much not a fair example for scipy since it wasn't 
designed to deal with data like this, and it shows in the results. Scipy 
assigns data to the correct cluster 84.7% of the time, compared with 99.1% 
for toroidal k-means and 99.95% for toroidal k-medians.