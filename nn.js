/*
* nn.js Tom Merchant 2017
* simple perceptron neural network implementation
*/

window.nn = {}
window.nn.activationFunctions = {}
window.nn.util = {}

/***
*Linear activation function
*@return wa
***/
window.nn.activationFunctions.identity = function(wa)
{
  return wa;
};

/***
*logistic sigmoid activation function
*@return 1/(1 + e^(-wa))
***/
window.nn.activationFunctions.logistic = function(wa)
{
  return 1/(1 + Math.exp(-wa));
};

/***
*Derivative of the logistic sigmoid activation function
***/
window.nn.activationFunctions.dlogistic = function(x)
{
    var f = window.nn.activationFunctions.logistic(x);
    return f*(1-f);
};

/***
*Tanh activation function
***/
window.nn.activationFunctions.tanh = Math.tanh;

/***
*Derivative of tanh
***/
window.nn.activationFunctions.dtanh = function(x)
{
    return 1 - Math.pow(Math.tanh(x), 2);
};

/***
*The Gaussian activation function
*@return e^(-wa^2)
***/
window.nn.activationFunctions.gaussian = function(wa)
{
  return Math.exp(-(wa*wa));
};

window.nn.activationFunctions.dgaussian = function(x)
{
    return -2*x*window.nn.activationFunctions.gaussian(x);
};

/***
*The rectified linear unit function
*@return 0 for wa < 0, or wa
***/
window.nn.activationFunctions.reLU = function(wa)
{
    return (wa > 0) * wa;
};

window.nn.activationFunctions.drelU = function()
{
    return (x > 1) * 1;
};

window.nn.activationFunctions.softplus = function(wa)
{
    return Math.ln(1+Math.exp(wa));
};

window.nn.activationFunctions.dsoftplus = window.nn.activationFunctions.logistic;

/***
*The bent identity function
*@return the bent identity
***/
window.nn.activationFunctions.bentIdentity = function(wa)
{
    return (Math.sqrt(wa*wa + 1) - 1)/2 + wa;
};

window.nn.activationFunctions.dbentIdentity = function(x)
{
    return x / (2*Math.sqrt(x*x + 1)) + 1;
}

/***
*A little pointless, basically just wraps Math.sin
***/
window.nn.activationFunctions.sinusoid = Math.sin;
window.nn.activationFunctions.dsinusoid = Math.cos;

/***
*Sinc function
*@return sinc(wa)
***/
window.nn.activationFunctions.sinc = function(wa)
{
    if(wa !== 0)
    {
        return Math.sin(wa)/wa;
    }
    return 1;
};

/***
*Derivative of the sinc function
***/
window.nn.activationFunctions.dsinc = function(x)
{
    if (x !== 0)
    {
        return Math.cos(x) / x - Math.sin(x) / (x*x);
    }
    return 0;
};

/***
*Leaky linear rectifier unit
*@return 0.01x for x < 1 else x
***/
window.nn.activationFunctions.lReLU = function(wa)
{
    if(wa < 1)
    {
        return 0.01*wa;
    }
    return wa;
};

window.nn.activationFunctions.dlReLU = function(x)
{
    if(x > 0)
    {
        return 1;
    }

    return 0.01;
};


window.nn.util.new2dArray = function(width, height, random)
{
    var array = [];

    for(var i = 0; i < width; i++)
    {
        array[i] = [];
        for(var j = 0; j < height; j++)
        {
            array[i][j] = 0 + random * (Math.random()-0.5);
        }
    }

    return array;
};

window.nn.util.newWeightMatrix = function(layers, width, inputs, outputs)
{
    var matrix = [];


    for(var layer = 0; layer < layers+1; layer++)
    {
        if(layer === 0)
        {
            matrix[0] = window.nn.util.newr2dArray(inputs, width, true);
        }
        else if(layer < layers)
        {
            matrix[layer] = window.nn.util.newr2dArray(width, width, true);
        }
        else
        {
            matrix[layers] = window.nn.util.newr2dArray(width, outputs, true);
        }
    }

    return matrix;
};

window.nn.newValueMatrix = function(layers, width, inputs, outputs)
{
    layers = [];

    for(var i = 0; i < layers+2; i++)
    {
        layers[i] = [];

        var s;

        if(i === 0)
        {
            s = inputs;
        }
        else if(i < layers)
        {
            s = width;
        }
        else
        {
            s = outputs;
        }

        for(var m = 0; m < s; m++)
        {
            layers[i][m] = 0;
        }
    }
    return layers;
};

/***
*Constructs a neural network based on the parameters
*
*@param [Number] inputs The number of input nodes
*@param [Number] layers The number of hidden layers
*@param [Number] width The nmber of nodes in each hidden layer
*@param [Number] outputs The number of output neurons
*@param [Function] phi The activation function to use
*@param [Function] d The derivative of the activation function
*@param [Number] alpha The learning rate
*@param [Boolean] notbias Whether to not use a bias input node
***/
window.nn.NNet = function(inputs, layers, width, outputs, phi, dphi, alpha, notbias)
{
    this.layers = layers;
    this.inputs = inputs+(1 * (!notbias));
    this.notbias = notbias;
    this.width = width;
    this.outputs = outputs;

    this.phi = phi || activationFunctions.identity;
    this.dphi = dphi || function(){return 1;};

    this.alpha = alpha || 0.7;

    this.weights = window.nn.util.newWeightMatrix(layers, width, this.inputs, outputs);
    this.values = window.nn.util.newValueMatrix(layers, width, this.inputs, outputs);
    return this;
};

this.nn.NNet.prototype.assertWeight = function(layer1, neuron1, neuron2)
{
    this.assertNeuron(layer1, neuron1);
    this.assertNeuron(layer1+1, neuron2);

};

window.nn.NNet.prototype.setWeight = function(layer1, neuron1, neuron2, weight)
{
    this.assertWeight(layer1, neuron1, neuron2);

    this.weights[layer1][neuron1][neuron2] = weight;
    this.weights[layer1][neuron2][neuron1] = weight;
};

window.nn.NNet.prototype.getWeight = function(layer1, neuron1, neuron2)
{
    this.assertWeight(layer1, neuron1, neuron2);

    return this.weights[layer1][neuron1][neuron2];
};

window.nn.NNet.prototype.assertLayer = function(layer)
{
    if(layer >= this.layers + 2)
    {
        throw "No such layer!";
    }
};

window.nn.NNet.prototype.assertNeuron = function(layer, neuron)
{
    this.assertLayer(layer);
    if(layer === 0 && neuron >= this.inputs || layer < this.layers && neuron >= width || neuron >= this.outputs)
    {
        throw "No such neuron";
    }
};

window.nn.NNet.prototype.getValue = function(layer, neuron)
{
    this.assertNeuron(layer, neuron);

    return this.values[layer][neuron];
};

window.nn.NNet.prototype.setValue = function(layer, neuron, value)
{
    this.assertNeuron(layer, neuron);

    this.values[layer][neuron] = value;
};

window.nn.NNet.prototype.setInputs = function(inputs)
{
    if(this.inputs.length === inputs.length)
    {
        this.values[0] = inputs;

        if(!this.notbias)
        {
            this.values[0].append(1);
        }
    }
    else
    {
        throw "Input number mismatch!";
    }
};

window.nn.NNet.prototype.getInputs = function()
{
    var inputs = this.values[0].slice(0);
    if(!this.notbias)
    {
        inputs.pop();
    }
    return inputs;
};

window.nn.NNet.prototype.getOutputs = function()
{
    return this.values[this.layers];
};

window.nn.NNet.prototype.processLayer = function(layer, inputs, outputs)
{
    var weightMatrix = this.weights[layer];

    for(var n2 = 0; n2 < inputs; n2++)
    {
        var wa = 0;

        for(var n1 = 0; n1 < outputs; n1++)
        {
            var val = this.getValue(layer, n1);
            var weight = this.getWeight(layer, n1, n2);

            wa += val * weight;
        }

        this.setValue(layer+1, n2, this.phi(wa));
    }
};

window.nn.NNet.prototype.setLossFunction = function(L, dL)
{
    this.L = L;
};

window.nn.NNet.prototype.L = function(expected, output)
{
    return 0.5 * Math.pow(expected - output, 2);
};

window.nn.NNet.prototype.dL = function(expected, output)
{
    return (output - expected);
};

window.nn.NNet.prototype.propagate = function()
{
    for(var layer = 0; layer < this.layers+2; layer++)
    {
        if(layer == 0)
        {
            this.processLayer(0, this.inputs, this.width);
        }
        else if(layer < this.layers)
        {
            this.processLayer(layer, this.width, this.width);
        }
        else
        {
            this.processLayer(layers+1, this.width, this.outputs);
        }
    }
}

window.nn.NNet.prototype.getNNodes = function(layer)
{
    this.assertLayer(layer);

    if(layer == 0)
    {
        return this.inputs;
    }
    else if(layer < this.layers)
    {
        return this.width;
    }
    else
    {
        return this.outputs
    }

}

window.nn.backPropagate = function(layer, dl, outputNeuron, expectedValue, matrix, _this)
{
    if(layer === _this.layers+1)
    {
        var deltaMatrix = window.nn.newWeightMatrix(_this.layers, _this.width, _this.inputs, _this.outputs);
        var L = _this.L(expectedValue, _this.getOutputs()[outputNeuron]);
        var dL = _this.dL(expectedValue, _this.getOutputs()[outputNeuron]);
        return backPropagate(layer - 1, dL, outputNeuron, expectedValue, newMatrix, _this);
    }
    else if(layer < _this.layers+1)
    {
        for(var node = 0; node < this.getNNodes(layer); node++)
        {
            //dL can be summarised as the error of the output node
            //dw gives us our contribution to the output node
            //dL * dw gives us our relative error for this node

            var dw = _this.getValue(layer, node) * _this.dphi(_this.getValue(layer, node) * _this.getWeight(layer, node, outputNeuron));
            var dL = dl * dw;

            var Dw = -_this.alpha * dL;

            //This bit of code is fairly ugly, but I don't want to add a paramter to the function so this is how we'll do it
            var wPtr = _this.weights;
            _this.weights = matrix;
            _this.setWeight(layer, node, outputNeuron, Dw);
            _this.weights = wPtr;

            var nm = backPropagate(layer-1, dL, node, 0, matrix, _this);

            if(node == this.getNNodes(layer)-1)
            {
                return nm;
            }
        }
    }
    else
    {
        //base case, ultimately we want to return the matrix of weight deltas we have constructed
        return matrix;
    }
}

window.nn.NNet.prototype.addMatrix = function(matrix)
{
    for(var i = 0; i < matrix.length; i++)
    {
        for(var j = 0; j < matrix[i].length; j++)
        {
            for(var k = 0; k < matrix[i][j].length; k++)
            {
                this.weights[i][j][k] += matrix[i][j][k];
            }
        }
    }
};

/***
 *Back propagation training
 *
 *@param dataset an array of object of form {inputs: [1, 2, etc...], outputs: [0, 1, 0, etc...]}
 ***/
window.nn.NNet.prototype.train = function(dataset)
{
    //The idea is to propagate the dataset values forward through the neural net
    //Once we have done that, the Loss function can be evaluated as a function of y
    //where y = a(x) where a is one path of our net and x is the set of inputs
    //We ultimately need to calculate the Loss function for each output and work out
    //the derivative of the Loss function with respect to specific weightings in our network

    //Error in output = (change in loss / change in output neuron) * dphi(input to output node)

    //Delta rule
    //dW = -alpha * dL * dphi(wt)
    //Change in loss with respect to any weight = phi(input*(loss of the neuron))

    for(var data in dataset)
    {
        this.setInputs(data.inputs);
        this.propagate();



        var matrices = [];

        for(var o = 0; o < this.outputs; o++)
        {
            matrices.push(window.nn.backPropagate(this.layers+1, 0, o, data.outputs[o], null, this));
        }

        //Now we simply add all the weights in all these matrices to the weight matrix
        for(var m = 0; m < matrices.length; m++)
        {
            this.addMatrix(matrices[m]);
        }
    }
}
