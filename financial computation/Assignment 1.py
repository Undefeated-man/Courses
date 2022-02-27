#!/usr/bin/env python
# coding: utf-8

#   

# <br />
# 
# **<font size=7> <center>Week 2 Assignment </center></font>**

# <font size=4> student_ID: <font color="red">**1930026174** </font></font>
# 
# <font size=4> Name: <font color="red">**Yuhan Zheng**</font></font>
# <br/><br/>

# In[1]:


import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns

from matplotlib import pyplot as plt
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')


# ### 1.1 Set up my github repository
# 

# ### 1.2 Basic Python

# In[2]:


# task 1
r1 = np.random.normal(loc=.05, scale=1, size=30)
r2 = np.random.normal(loc=.05, scale=1, size=30)
r3 = np.random.normal(loc=.05, scale=1, size=30)


# In[3]:


# task 2
print("The r1 mean is:", r1.mean())
print("The r1 std is:", r1.std())
print("The r2 mean is:", r2.mean())
print("The r2 std is:", r2.std())
print("The r3 mean is:", r3.mean())
print("The r3 std is:", r3.std())


# In[4]:


# Task 3
for i in [r1, r2, r3]:
    if i.mean() <0:
        print("The mean is smaller than 0")


# In[5]:


# Task 4
def simulate(n:int, mu:float, var:float):
    tmp = []
    for i in range(n):
        tmp.append(np.random.normal(loc=mu, scale=var, size=30))
    
    for series in tmp:
        if series.mean() <0:
            return "The mean is negative!"


# Test
simulate(5, mu=.05, var=1)


# ## 2 Numerical Computing

# ### 2.1 Newton's Method

# In[6]:


# Task 5 - Newton method
def myfunc(x):
    y = (1 + 0.5 * x)**5
    return


def numerical_grad(func, x, dx = 0.00001) :
    """
        Impletementing the numerical gradient descent.
    """
    dy = func(x+dx) - func(x)
    ngrad = dy/dx
    return ngrad


def newton_method(func , func_value = 1.1 , x=-1, max_iteration = 1000000 , max_err = 0.000001 ):
    """
        Func:
            Find the x of the corresponding function's specific function value
            
        Args:
            func: a defined function
            func_value: the specified function value (y)
            x: the initial guess
            max_iteration: the maximum iteration allowed
            max_err: the thredshold of the error
    """
    for trial in range( max_iteration ):
        error = func(x) - func_value
        if abs( error ) < max_err :
            y = func(x)
            print(" Iteration {}: f = {}, x = {}".format(trial , y, x))
            return x
        else :
            grad = numerical_grad(func, x)
            x = x - error/grad
            y = func(x)
            print(" Iteration {}: f = {}, x = {}".format(trial , y, x))
    raise ValueError(" Max iteration reached ")
    print(" Max Iteration {}: f = {}, x = {}".format(trial , y, x))

    
def test(x):
    return np.log(x)

# test
if __name__ == "__main__":
    newton_method(test, x=2)


# ### 2.2 Gradient Descent

# In[7]:


# Task 6
def test(x):
    """
        The test function
    """
    return x**2 - 2*x + 3


def f(func , x=1.1, max_iteration = 1000000 , max_err = .000001, lr = .01, text_track=False):
    """
        Func:
            Find the minimum point of the function.
            
        Args:
            func: a defined function
            x: the initial guess
            max_iteration: the maximum iteration allowed
            max_err: the thredshold of the error
            lr: the learning rate. default to be 0.01
            text_track: whether showing the text track information (default to be false)
            
        Returns:
            the final computed result of x
    """
    tmp = [x]
    for trial in range( max_iteration ):
        if len(tmp) > 1:
            df = tmp[-2] - x
        else:
            df = 10  # ensure that it can continue to process
        if abs( df ) < max_err :
            y = func(x)
            print(" Iteration {}: y = {}, x = {}".format(trial , y, x))
            plt.figure(figsize=(10,6))
            plt.title('The track of x', {"size": 20})
            plt.plot(np.array(tmp), linewidth=2.5, alpha=.5)
            plt.xlabel('Iteration', fontsize=18)
            plt.ylabel('Value', fontsize=18)
            return x
        else :
            grad = numerical_grad(func, x)
            x = x - lr*grad
            tmp.append(x)
            y = func(x)
            if text_track:
                print(" Iteration {}: y = {}, x = {}".format(trial , y, x))
    raise ValueError(" Max iteration reached ")
    print(" Max Iteration {}: y = {}, x = {}".format(trial , y, x))
    
# test
if __name__ == "__main__":
    f(test, x=0)


# In[ ]:




