# TEST
import numpy as np
import config

def add_padding_to_additional_input(input, timesteps = config.TIMESTEPS, projection_time = config.PROJECTION_TIME):
    """
    Padding for additional_input. To predict the net profit at timestep t the neural network needs additional_input at timesteps
    t-1, t-2, ..., t-(timesteps+1). So, e.g. to predict net profit at timestep 1, we need padding and use the additional_input at 
    t=0 (timesteps+1) times.
    """
    for i in range(timesteps):

        indices = np.arange(0, input.size + 1, projection_time + i)

        input = np.insert(input, indices[:-1], input[indices[:-1]])

    return input

def add_padding_to_input(input, timesteps = config.TIMESTEPS, projection_time = config.PROJECTION_TIME):
    """
    Padding for regular input. To predict the net profit at timestep t the neural network needs inputs at timesteps
    t, t-1, ..., t-timesteps. So, e.g. to predict net profit at timestep 1, we need padding and use the inputs at 
    t=1, plus t=0 timesteps times.
    """

    for i in range(timesteps - 1):

        indices = np.arange(0, input.shape[0] + 1, projection_time + i + 1)

        input = np.insert(input, indices[:-1], input[indices[:-1]])

    return input


x = np.array(range(120))
print('x: ', x)
print('len(x): ', x.size)

# multiple inserts at once
indices = np.arange(0, x.size + 1, 60)
print(indices)

# Remove last value of subsequence
x = np.delete(x, indices[1:] - 1)
print('x after delete: ', x)
print('length after delete: ', x.size)

x = add_padding_to_input(x)
print('x using add_padding: ', x)

a = np.array([[1,1,1,1,1], [2,2,2,2,2], [3,3,3,3,3], [4,4,4,4,4]])
print(a)
print(a.shape)

ind = np.arange(0, a.shape[0], 2)
print(ind)

a = np.insert(a, ind, a[ind,:], axis=0)
print(a)

# Adjust indices (now 59 entries per scenario)
# indices = np.arange(0, x.size + 1, 59)

# rep = np.repeat(x[indices[:-1]], config.TIMESTEPS)
# print('repeat: ')
# print(rep)

# rep_list = [np.repeat(i, 2) for i in x[indices[:-1]]]
# print('repeat list: ', rep_list)
# rep_list = np.array(rep_list)
# print('rep_list as np: ', rep_list)
# print(rep_list[:,0])

# print('indices[:-1]: ', indices[:-1])
# # z = [np.insert(x, indices[:-1], rep_list[i]) for i in range(2)]
# z = np.insert(x, indices[:-1], x[indices[:-1]])
# print('z: ', z)
# z = np.insert(x, indices[:-1], rep_list)
# # z = np.insert(x, 0, np.repeat(0, 5))

# print('x after insert: ', z)
# print('len(z): ', z.size)
