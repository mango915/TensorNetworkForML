def psi(x):
    x = np.array((np.sin(np.pi*x/2),np.cos(np.pi*x/2)))
    return np.transpose(x, [1,2,0])
X = np.random.random((B, N))
X = psi(X)
print(X.shape)
print(X.sum())
f = net.forward(X)
f2 = net.forward(X)
print(f.elem)
print(f2.elem)


svd_accs = []
for i in range(self.N-1,0,-1):
    print('i: ',i)
    B = contract(self.As[i-1], self.As[i], "right", "left")
    print("B: ", B)
    # reconstruct optimized network tensors
    B.aggregate(axes_names=['d'+str(i-1),'left','l'], new_ax_name='i')
    B.aggregate(axes_names=['d'+str(i),'right'], new_ax_name='j')
    B.transpose(['i','j'])
    self.As[i-1], self.As[i], svd_acc = tensor_svd(B, inverse=True)
    svd_accs.append(svd_acc)
    print('self.As[%d]: '%(i-1), self.As[i-1])
print('self.As[0] ', self.As[0])
svd_accs = np.array(svd_accs)
tot_acc = np.prod(svd_accs)
print('Final accuracy SVD: ', tot_acc)
self.l_pos = 0

        #B = contract(self.As[-1], self.As[0], "right", "left")
        # reconstruct optimized network tensors
        #B.aggregate(axes_names=['d'+str(self.l_pos),'left'], new_ax_name='i')
        #B.aggregate(axes_names=['d0','right','l'], new_ax_name='j')
        #B.transpose(['i','j'])
        #self.As[-1], self.As[0] = tensor_svd(B)