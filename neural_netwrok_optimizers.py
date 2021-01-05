
######## optimizers ##################3
########################################


name = [ 0 for i in range(opt_n)]
optimizer = [ 0 for i in range(opt_n)]
closure_list = [ 0 for i in range(opt_n)]
model = [0 for i in range(opt_n)] # list of models to be used with optimizers

for opt_i in opt_list:
    # Use the nn package to define a model for each optimizer
    model[opt_i] = copy.deepcopy(model_o)


###########  ALR-GDc #############
n = 0
if n in opt_list:
    name[0] = "ALR-GDc"
    from neural_network_ALR_GDc import ALR_GDc
    drift=True
    optimizer[0] = ALR_GDc(model[0].parameters(), lr=lr[0], min_step_r=2**5, max_step_r=2**5 )
    #### defining the function to reevalute function and gradient if needed
    def closure():
        optimizer[0].np_to_params()
        optimizer[0].zero_grad()
        y_pred = model[0](x_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer[0].params_to_np()
        return loss.item()
    closure = torch.enable_grad()(closure)
    closure_list[0] = closure


###########  ALR-GD #############
n = 1
if n in opt_list:
    name[1] = "ALR-GD"
    from neural_network_ALR_GD import ALR_GD
    drift=True
    optimizer[1] = ALR_GD(model[1].parameters(), lr=lr[1])
    #### defining the function to reevalute function and gradient if needed
    def closure():
        optimizer[1].np_to_params()
        optimizer[1].zero_grad()
        y_pred = model[1](x_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer[1].params_to_np()
        return loss.item()
    closure = torch.enable_grad()(closure)
    closure_list[1] = closure


# ###### ADAM ##################
n = 2
if n in opt_list:
    name[2] = "ADAM"
    from neural_network_ADAM import adam
    optimizer[2] = adam(model[2].parameters(), lr=lr[2])
    #### defining the function to reevalute function and gradient if needed
    def closure():
        optimizer[2].np_to_params()
        optimizer[2].zero_grad()
        y_pred = model[2](x_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer[2].params_to_np()
        return loss.item()
    closure = torch.enable_grad()(closure)
    closure_list[2] = closure



# ###### GDM ##################
n = 3
if n in opt_list:
    name[3] = "GDM"
    from neural_network_GDM import gdm
    optimizer[3] = gdm(model[3].parameters(), lr=lr[3])
    #### defining the function to reevalute function and gradient if needed
    def closure():
        optimizer[3].np_to_params()
        optimizer[3].zero_grad()
        y_pred = model[3](x_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer[3].params_to_np()
        return loss.item()
    closure = torch.enable_grad()(closure)
    closure_list[3] = closure




# # ###### GDM ##################
# n = 3
# name[3] = "GDM"
# momentum = 0.9
# from neural_network_GDM import gdm
# optimizer[3] = gdm(model[3].parameters(), lr=lr[3], momentum = 0.9)
# # optimizer[3] = torch.optim.SGD(model[3].parameters(), lr=lr[3], momentum=momentum)
# #### defining the function to reevalute function and gradient if needed
# def closure():
#     optimizer[3].np_to_params()
#     optimizer[3].zero_grad()
#     y_pred = model[3](x_train)
#     loss = loss_fn(y_pred, y_train)
#     loss.backward()
#     optimizer[3].params_to_np()
#     return loss.item()
# closure = torch.enable_grad()(closure)
# closure_list[3] = closure


###########  ALR-GDMd #############
n = 4
if n in opt_list:
    name[4] = "ALR-GDMd"
    from neural_network_ALR_GDM import abgd_vmd
    momentum = 0.7
    drift = True
    optimizer[4] = abgd_vmd(model[4].parameters(), lr=lr[4], momentum=momentum, drift=drift)
    #### defining the function to reevalute function and gradient if needed
    def closure():
        optimizer[4].np_to_params()
        optimizer[4].zero_grad()
        y_pred = model[4](x_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer[4].params_to_np()
        return loss.item()
    closure = torch.enable_grad()(closure)
    closure_list[4] = closure


# ###### ALR-ADAM ##################
n = 5
if n in opt_list:
    name[5] = "ALR-ADAM"
    from neural_network_ALR_ADAM import ALR_ADAM
    optimizer[5] = ALR_ADAM(model[5].parameters(), lr=lr[5])
    #### defining the function to reevalute function and gradient if needed
    def closure():
        optimizer[5].np_to_params()
        optimizer[5].zero_grad()
        y_pred = model[5](x_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer[5].params_to_np()
        return loss.item()
    closure = torch.enable_grad()(closure)
    closure_list[5] = closure


# ###### ALR-GDM ##################
n = 6
if n in opt_list:
    name[6] = "ALR-GDM"
    from neural_network_ALR_GDM import ALR_GDM
    optimizer[6] = ALR_GDM(model[6].parameters(), lr=lr[6])
    #### defining the function to reevalute function and gradient if needed
    def closure():
        optimizer[6].np_to_params()
        optimizer[6].zero_grad()
        y_pred = model[6](x_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer[6].params_to_np()
        return loss.item()
    closure = torch.enable_grad()(closure)
    closure_list[6] = closure

#
# # ###### ALR-ADAM2_copy ##################
# n = 7
# name[7] = "ALR-ADAM2_copy"
# from neural_network_ALR-ADAM2_copy import abgd_cm2_copy
# optimizer[7] = abgd_cm2_copy(model[7].parameters(), lr=lr[7])
# #### defining the function to reevalute function and gradient if needed
# def closure():
#     optimizer[7].np_to_params()
#     optimizer[7].zero_grad()
#     y_pred = model[7](x_train)
#     loss = loss_fn(y_pred, y_train)
#     loss.backward()
#     optimizer[7].params_to_np()
#     return loss.item()
# closure = torch.enable_grad()(closure)
# closure_list[7] = closure
#
#
# # ###### ALR-GDM2 ##################
# n = 8
# name[8] = "ALR-GDM2"
# from neural_network_ALR-GDM2 import abgd_vm
# optimizer[8] = abgd_vm_copy(model[8].parameters(), lr=lr[8])
# #### defining the function to reevalute function and gradient if needed
# def closure():
#     optimizer[8].np_to_params()
#     optimizer[8].zero_grad()
#     y_pred = model[8](x_train)
#     loss = loss_fn(y_pred, y_train)
#     loss.backward()
#     optimizer[8].params_to_np()
#     return loss.item()
# closure = torch.enable_grad()(closure)
# closure_list[8] = closure
