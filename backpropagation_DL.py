import random
import math

class Neuron:
    def __init__(self, bia):
        self.bia = bia
        self.weights = []
        
    def WX_plus_b(self):
        total = 0
        for i in range(len(self.inputs)):
            total += self.inputs[i] * self.weights[i]
        return total + self.bia

    def sigmoid(self, inputs_plus):
        return 1 / (1 + math.exp(-inputs_plus))

    def from_X_to_H(self, inputs):
        self.inputs = inputs
        self.output = self.sigmoid(self.WX_plus_b())
        return self.output

    def one_error(self, target_output):
        return 0.5 * (target_output - self.output) **2
    
    def one_error_pd_O(self, target_output):
        return -(target_output - self.output)
    
    def pd_sigmoid(self):#sigmoid(1-sigmoid),H(1-H)，
        #这里我想定一个params，但后面会出错，待会回来再弄
        return self.output * (1 - self.output)

    def one_error_pd_vs_o(self, target_output):
        return self.one_error_pd_O(target_output) * self.pd_sigmoid();
    
    def h_pd_w(self, index):
        return self.inputs[index]
#	def one_error_pd_w(self)
#		return self.one_error_pd_vs_o(target_output)* self.h_pd_w
		
class Layer:#经过本层计算后得到本层各神经元输出的元组
    def __init__(self, num_neuron, bia):

        self.bia = bia if bia else random.random()

        self.neurons = []#前馈网络
        for i in range(num_neuron):
            self.neurons.append(Neuron(self.bia))
  
    def from_X_H_list(self, inputs):#这一步将Neuron的计算结果传入self.neurons[]中生成list了
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.from_X_to_H(inputs))
        return outputs

		
class Network:
    Learn_Rate = 0.3

    def __init__(self, num_inputs, num_hidden, num_output, Wi = None, b1 = None, Wh = None, b2 = None): 
	
        self.num_inputs = num_inputs

        self.H_layer = Layer(num_hidden, b1)
        self.O_layer = Layer(num_output, b2)

        self.initial_Wi(Wi)#未赋予实例的方法被引用，就用self.函数名 的方式
        self.initial_Wh(Wh)

    def initial_Wi(self, Wi):
        weight_num = 0
        for h in range(len(self.H_layer.neurons)):#Wi是个h*i的权重矩阵
            for i in range(self.num_inputs):
                if not Wi:
                    self.H_layer.neurons[h].weights.append(random.random())
                else:
                    self.H_layer.neurons[h].weights.append(Wi[weight_num])
                weight_num += 1

    def initial_Wh(self, Wh):#一个o*h的权重矩阵
        weight_num = 0
        for o in range(len(self.O_layer.neurons)):
            for h in range(len(self.H_layer.neurons)):
                if not Wh:
                    self.O_layer.neurons[o].weights.append(random.random())
                else:
                    self.O_layer.neurons[o].weights.append(Wh[weight_num])
                weight_num += 1

    def from_V_H_O(self, inputs):#这个程序把它定死就是3层网络
        H = self.H_layer.from_X_H_list(inputs)
        return self.O_layer.from_X_H_list(H)
    
    def error_sum(self, training_sets):
        total_error = 0
        for t in range(len(training_sets)):
            training_inputs, training_outputs = training_sets[t]
            self.from_V_H_O(training_inputs)
            for o in range(len(training_outputs)):
                total_error += self.O_layer.neurons[o].one_error(training_outputs[o])
        return total_error

    def train(self, training_inputs, training_outputs):
	
        self.from_V_H_O(training_inputs)
		
        error_sum_pd_o = [0] * len(self.O_layer.neurons)
        for o in range(len(self.O_layer.neurons)):
            error_sum_pd_o[o] = self.O_layer.neurons[o].one_error_pd_vs_o(training_outputs[o])
			
		#上面self.O_layer.neurons[o]是因为语句self.neurons.append(Neuron(self.bia))
		#把neurons[o]变成了Neuron父类的实例集合
		
        error_sum_pd_h = [0] * len(self.H_layer.neurons)
        for h in range(len(self.H_layer.neurons)):

            error_summer_pd_H = 0
            for o in range(len(self.O_layer.neurons)):
                error_summer_pd_H += error_sum_pd_o[o] * self.O_layer.neurons[o].weights[h]
				#∂E/∂H=（∂e1/∂H+∂e2/∂H）

            error_sum_pd_h[h] = error_summer_pd_H * self.H_layer.neurons[h].pd_sigmoid()
			#∂E/∂h=（∂e1/∂H+∂e2/∂H）*∂H/∂h

        for o in range(len(self.O_layer.neurons)):
            for w_ho in range(len(self.O_layer.neurons[o].weights)):

                error_sum_pd_w = error_sum_pd_o[o] * self.O_layer.neurons[o].h_pd_w(w_ho)#特指H-o之间的权重

                self.O_layer.neurons[o].weights[w_ho] -= self.Learn_Rate * error_sum_pd_w#修改权重

				
        for h in range(len(self.H_layer.neurons)):
            for w_ih in range(len(self.H_layer.neurons[h].weights)):

                error_sum_pd_w = error_sum_pd_h[h] * self.H_layer.neurons[h].h_pd_w(w_ih)#特指v-h之间的权重

                self.H_layer.neurons[h].weights[w_ih] -= self.Learn_Rate * error_sum_pd_w#修改权重

    

		
		




nn = Network(2, 3, 2, Wi=[0.15, 0.2, 0.25, 0.3, 0.25, 0.3], b1=0.35, Wh=[0.4, 0.45, 0.5, 0.55, 0.5, 0.55], b2=0.6)
for i in range(10):
    nn.train([0.05, 0.1], [0.01, 0.99])
    print(i, round(nn.error_sum([[[0.05, 0.1], [0.01, 0.99]]]), 5))
#注意输出为error_sum,因为我们的目标就是误差最小，或者说误差为
#网络2,3,2结构从17步开始就比2,2,2结构收敛快了