# cow
7.26日：
文章暂时没有发，所以只放了一部分的代码，后续进行补充。此处记录一些问题。

在计算模型大小时，inputsze大小计算有问题，使用如下代码，替换torchsummary文件中100~103行左右代码。即将

    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size
替换为

    total_input_size = abs(np.sum([np.prod(in_tuple) for in_tuple in input_size]) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

模型inputsize正确，网络模型大小正确。

