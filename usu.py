import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.colors import ListedColormap 
import numpy as np
import torch 
import matplotlib.markers as mmarkers
import warnings


"""

2. a 3. cviceni

"""

def draw(x, y, title, x_label, y_label, *args):
    """
    args[0] = theta
    args[1] = valueX
    args[2] = valueY
    """
    figure(figsize=(10, 6), dpi=90)
    
    size = np.size(x)
        
    if len(args) >= 1:
        theta = args[0]
        x_line = np.linspace(np.floor(np.min(x)), np.floor(np.max(x)) + 1, size)
        y_pred = compute_prediction(x_line, theta)
        y_pred = np.reshape(y_pred, size)
        plt.plot(x_line, y_pred, linewidth=2, color='green')
        
        if len(args) == 3: 
            valueX = float(args[1])
            valueY = float(args[2])
            
            if(valueX >= np.min(x_line) and valueX <= np.max(x_line)):
                plt.axvline(x=valueX,color='red', linestyle=':')

            if (valueY >= np.min(y_pred) and valueY <= np.max(y_pred)):
                plt.axhline(y=valueY, color='red', linestyle=':')
                plt.plot(valueX, valueY, marker='.', color='red', markersize=8)
            print(f"x : {valueX:.3f}, y : {valueY:.3f}")
        

    plt.scatter(x, y, s=26, marker='x', color='blue')
    
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
    
def compute_prediction(x, theta):
    y_pred = np.zeros(x.shape)
    for i in range(theta.size):
        y_pred += theta[i] * x**i
    return y_pred


"""
4. a 5. cviceni
"""

def draw2d(train_data, train_targets, test_data, test_targets, etalons=None, title='Data', x_label='x', y_label='y'):
    color_point_test = ListedColormap(['#e60000', '#00802b', '#004a99', '#ff9900', ])
    color_point_train = ListedColormap(['#ff6666', '#70db70', '#66b0ff', '#ffc266', ])
    
    figure(figsize=(12, 8), dpi=90)

    plt.scatter(train_data.T[0], train_data.T[1], s=40, marker='o', c=train_targets, cmap=color_point_train, label="trainData")
    plt.scatter(test_data.T[0], test_data.T[1], s=40, marker='o', c=test_targets, cmap=color_point_test, edgecolors='black', label="testData")

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
    
    
"""
6., 7. cviceni
"""
def checkSigmoid(sigmoid):
    m = torch.nn.Sigmoid()
    inp = torch.randn(2,3)
    output = m(inp)
    output2 = sigmoid(inp)
    return np.allclose(output2,np.array(output))

def checkSoftmax(softmax):
    m = torch.nn.Softmax(dim=0)
    inp = torch.randn(2,3)
    outputtorch = m(inp)
    output = softmax(np.array(inp))
    return np.allclose(outputtorch,np.array(output))


"""def mscatter(data, classes=None, theta=None):
    fig = plt.figure(figsize=(10,6),dpi=90)
    ax = plt.gca()
    
    data = data.T
    cmap = ListedColormap(['#ff6666', '#66ff99', '#66b0ff', '#ffc266', ])
    markers = ['X','*', 'd', 's', 'o']
    
    sc = ax.scatter(data[0], data[1], cmap=cmap, c=classes, s=45)
    
    if (classes is not None) and (len(classes)==len(data[0])):
        paths = []
        
        for idx in classes:
            if isinstance(markers[int(idx)], mmarkers.MarkerStyle):
                marker_obj = markers[int(idx)]
            else:
                marker_obj = mmarkers.MarkerStyle(markers[int(idx)])
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
        if(theta is not None):
            u = np.linspace(np.floor(np.min(data)), np.floor(np.max(data)) + 1, 10)
            v = np.linspace(np.floor(np.min(data)), np.floor(np.max(data)) + 1, 10)
            z = np.zeros([10, 10]);
            for i in range(10):
                for j in range(10):
                    z[i,j] = [1, u[i], v[j]] @ theta
            CS = ax.contour(u, v, z.T, [-2,0,2])
            ax.clabel(CS, inline=True, fontsize=10)
    plt.show()"""
    
def drawSoftmax(data, classes, theta=None, softmax=None, title='Data', x_label='x1', y_label='x2'):

    #red,green,blue, fial, orange
    color_point = ['#ff6666', '#66ff99', '#66b0ff','#633974', '#ffc266', ]
    #red,green,blue, fial, orange, orange
    color_background = ['#ffcccc', '#ccffdd', '#cce5ff', '#c9aad5', '#c9aad5', '#ffebcc',]    
    markers = ['X','*', 'd', 's', 'o']

    cmap_point = ListedColormap(color_point)
    cmap_background = ListedColormap(color_background)

    figure(figsize=(10, 6), dpi=90)
    x1 = data.T[0]
    x2 = data.T[1]
 
    if(theta is not None and softmax is None):
        ax = plt.gca()
        u = np.linspace(np.floor(np.min(data)), np.floor(np.max(data)) + 1, 10)
        v = np.linspace(np.floor(np.min(data)), np.floor(np.max(data)) + 1, 10)
        z = np.zeros([10, 10]);
        for i in range(10):
            for j in range(10):
                z[i,j] = [1, u[i], v[j]] @ theta
        CS = ax.contour(u, v, z.T, [-2,0,2])
        ax.clabel(CS, inline=True, fontsize=10)
    
    if (theta is not None and softmax is not None):
        #krivky
        ax = plt.gca()
        u = np.linspace(np.floor(np.min(data)), np.floor(np.max(data)) + 1, 250)
        v = np.linspace(np.floor(np.min(data)), np.floor(np.max(data)) + 1, 250)
        z = np.zeros([250, 250])
        #
        # for k in range(np.size(np.unique(classes))):
        #     for i in range(250):
        #         for j in range(250):
        #             z[i,j] = ([1, u[i], v[j]] @ theta[:,k]).T
        #     CS = ax.contour(u, v, z.T, [0], colors='k')
        #     ax.clabel(CS, inline=True, fontsize=10)
        
        #background
        z = np.zeros([250, 250,np.size(np.unique(classes))])
        
        for i in range(250):
            for j in range(250):
                z[i,j,:] = (softmax([1, u[i], v[j]] @ theta).T).T
        
        idx = np.argmax(z,axis=2)
        for k in range(np.size(np.unique(classes))):
           CS = ax.contourf(u, v, idx.T, cmap=cmap_background)

    sc = plt.scatter(x1, x2, s=20, c=classes.T, cmap=cmap_point)
    paths = []

    for idx in classes:
        if isinstance(markers[int(idx)], mmarkers.MarkerStyle):
            marker_obj = markers[int(idx)]
        else:
            marker_obj = mmarkers.MarkerStyle(markers[int(idx)])
        path = marker_obj.get_path().transformed(marker_obj.get_transform())
        paths.append(path)
    sc.set_paths(paths)    

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
    
    
"""
8. a 9. cviceni
"""

def drawSVM(data, classes, theta=None, title='Data', x_label='x1', y_label='x2'):
    import logging
    logging.getLogger().setLevel(logging.CRITICAL)
    #red,green,blue, fial, orange
    color_point = ['#ff6666', '#66ff99', '#66b0ff','#633974', '#ffc266', ]
    #red,green,blue, fial, orange, orange
    color_background = ['#ffcccc', '#ccffdd', '#cce5ff', '#c9aad5', '#c9aad5', '#ffebcc',]    
    markers = ['X','*', 'd', 's', 'o']
    #print(f"classes: \n \t 0: red \n\t 1: green")
    cmap_point = ListedColormap(color_point)
    cmap_background = ListedColormap(color_background)

    figure(figsize=(10, 6), dpi=90)
    x1 = data.T[0]
    x2 = data.T[1]
    
    if (theta is not None):
        #krivky
        ax = plt.gca()
        u = np.linspace(np.floor(np.min(data)), np.floor(np.max(data)) + 1, 250)
        v = np.linspace(np.floor(np.min(data)), np.floor(np.max(data)) + 1, 250)
        z = np.zeros([250, 250])
        #
        # for k in range(np.size(np.unique(classes))):
        #     for i in range(250):
        #         for j in range(250):
        #             z[i,j] = ([1, u[i], v[j]] @ theta[:,k]).T
        #     CS = ax.contour(u, v, z.T, [0], colors='k')
        #     ax.clabel(CS, inline=True, fontsize=10)
        
        #background
        z = np.zeros([250, 250,np.size(np.unique(classes))])
        
        for i in range(250):
            for j in range(250):
                z[i,j,:] = (([1, u[i], v[j]] @ theta).T).T
        
        idx = np.argmax(z,axis=2)
        for k in range(np.size(np.unique(classes))):
            CS = ax.contourf(u, v, idx.T, cmap=cmap_background)

    sc = plt.scatter(x1, x2, s=20, c=classes.T, cmap=cmap_point)
    paths = []

    legend1 = ax.legend(*sc.legend_elements(),
                    loc="lower left", title="Classes")
    ax.add_artist(legend1)
    for idx in classes:
        if isinstance(markers[int(idx)], mmarkers.MarkerStyle):
            marker_obj = markers[int(idx)]
        else:
            marker_obj = mmarkers.MarkerStyle(markers[int(idx)])
        path = marker_obj.get_path().transformed(marker_obj.get_transform())
        paths.append(path)
    sc.set_paths(paths)    

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()