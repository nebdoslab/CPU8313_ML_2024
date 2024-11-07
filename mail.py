import math

## H(X) = -Σ p(x) * log2(p(x))
def entropy(node:list) -> float:
    total = sum(node)
    e = 0
    for i in node:
        if i == 0: continue
        p = i / total
        e -= -p * math.log2(p)
        print(f'p({i}/{total}): {p}, e: {e}')
    return e


## out_{h1} = \frac{1}{1+e^{-net_{h1}}}
def sigmoid(net:float) -> float:
    return 1 / (1 + math.exp(-net))





def fwd_pass():
    t1 = 0.01
    t2 = 0.99

    ## Forward pass
    h1_act = 0.8 * 0.9 + -0.4 * 0.3 + 0.1 * 1
    print(f'h1: {h1_act}')
    h1 = sigmoid(h1_act)
    print(f'h1: {h1}')

    h2_act = -0.2 * 0.9 + 0.3 * 0.3 + 0.6 * 1
    print(f'h2: {h2_act}')
    h2 = sigmoid(h2_act)
    print(f'h2: {h2}')

    o1_act = -0.7 * h1 + -0.3 * h2 + 0.2 * 1
    print(f'o1: {o1_act}')
    o1 = sigmoid(o1_act)
    print(f'o1: {o1}')

    o2_act = 0.5 * h1 + 0.6 * h2 + 0.4 * 1
    print(f'o2: {o2_act}')
    o2 = sigmoid(o2_act)
    print(f'o2: {o2}')

    e1 = (t1 - o1) ** 2
    print(f'e1: {e1}')
    e2 = (t2 - o2) ** 2
    print(f'e2: {e2}')
    e_total = (e1 + e2) / 2
    print(f'e_total: {e_total}')


    ### Backward pass
    # d_et_wrt_o1 = (o1 - t1) ## derivative of total error wrt o1
    # print(f'd_et_wrt_o1: {d_et_wrt_o1}')

    # d_o1_wrt_o1_act = o1 * (1 - o1) ## derivative of o1 wrt o1_act
    # print(f'd of o1 wrt to activation: {d_o1_wrt_o1_act}')



###
###
if __name__ == '__main__':

    et_o1 = 0.3781314542973173
    o1_net1 = 0.23748542848236678

    et_o2 = -0.23806151280180476
    o2_net2 = 0.18652699866828482

    net_o1_out_h1 = -0.7
    net_o1_out_h2 = -0.3
    net_o2_out_h1 = 0.5
    net_o2_out_h2 = 0.6

    out_h1_net_h1 = 0.22171287329310904
    out_h2_net_h2 = 0.2344233439307613

    net_h1_w1 = 0.9
    net_h1_w2 = 0.3
    net_h2_w3 = 0.9
    net_h2_w4 = 0.3

    w1 = (et_o1 * o1_net1 * net_o1_out_h1 + et_o2 * o2_net2 * net_o2_out_h1) * out_h1_net_h1 * net_h1_w1
    print(f'w1: {w1}')
    w1_adj = 0.8 - 0.5 * w1
    print(f'w1_adj: {w1_adj}')

    w2 = (et_o1 * o1_net1 * net_o1_out_h1 + et_o2 * o2_net2 * net_o2_out_h1) * out_h1_net_h1 * net_h1_w2
    print(f'w2: {w2}')
    w2_adj = (-0.4) - 0.5 * w2
    print(f'w2_adj: {w2_adj}')

    w3 = (et_o1 * o1_net1 * net_o1_out_h2 + et_o2 * o2_net2 * net_o2_out_h2) * out_h2_net_h2 * net_h2_w3
    print(f'w3: {w3}')
    w3_adj = (-0.2) - 0.5 * w3
    print(f'w3_adj: {w3_adj}')

    w4 = (et_o1 * o1_net1 * net_o1_out_h2 + et_o2 * o2_net2 * net_o2_out_h2) * out_h2_net_h2 * net_h2_w4
    print(f'w4: {w4}')
    w4_adj = 0.3 - 0.5 * w4
    print(f'w4_adj: {w4_adj}')

    b1 = (et_o1 * o1_net1 * net_o1_out_h1 + et_o2 * o2_net2 * net_o2_out_h1) * out_h1_net_h1
    print(f'b1: {b1}')
    b1_adj = 0.1 - 0.5 * b1
    print(f'b1_adj: {b1_adj}')

    b2 = (et_o1 * o1_net1 * net_o1_out_h2 + et_o2 * o2_net2 * net_o2_out_h2) * out_h2_net_h2
    print(f'b2: {b2}')
    b2_adj = 0.6 - 0.5 * b2
    print(f'b2_adj: {b2_adj}')
    
  
