
window.nn.activationFunctions = {
  /***
  *Linear activation function
  *@return wa
  ***/
  identity: function(wa)
  {
      return wa;
  },
  /***
  *logistic sigmoid activation function
  *@return 1/(1 + e^(-wa))
  ***/
  logistic: function(wa)
  {
      return 1/(1 + Math.exp(-wa));
  },
  dlogistic: function(x)
  {
      var f = window.nn.activationFunctions.logistic(x);
      return f*(1-f);  
  },
  /***
  *Tanh activation function
  *@return tanh(wa)
  ***/
  tanh: function(wa)
  {
      return Math.tanh(wa);
  },
  dtanh: function(x)
  {
      return 1 - Math.pow(Math.tanh(x), 2);
  },
  /***
  *The Gaussian activation function
  *@return e^(-wa^2)
  ***/
  gaussian: function(wa)
  {
      return Math.exp(-(wa*wa));
  },
  dgaussian: function(x)
  {
      return -2*x*window.nn.activationFunctions.gaussian(x);
  },
  /***
   *The rectified linear unit function
   *@return 0 for wa < 0, or wa
   ***/
  reLU: function(wa)
  {
      return (wa > 0) * wa;
  },
  drelU: function()
  {
      return (x > 1) * 1;
  },
  softplus: function(wa)
  {
      return Math.ln(1+Math.exp(wa));
  },
  
  dsoftplus: window.nn.activationFunctions.logistic,
  
  /***
   *The bent identity function
   *@return the bent identity
   ***/
  bentIdentity: function(wa)
  {
      return (Math.sqrt(wa*wa + 1) - 1)/2 + wa;
  },
  
  dbentIdentity: function(x)
  {
      return x / (2*Math.sqrt(x*x + 1)) + 1;
  },
  
  /***
   *A little pointless, basically just wraps Math.sin
   ***/
  sinusoid: Math.sin,
  dsinusoid: Math.cos,
  
  /***
   *Sinc function
   *@return sinc(wa)
   ***/
  sinc: function(wa)
  {
      if(wa !== 0)
      {
          return Math.sin(wa)/wa;
      }
      return 1;
  },
  dsinc: function(x)
  {
      if (x !== 0)
      {
          return Math.cos(x) / x - Math.sin(x) / (x*x);
      }
      return 0;
  },
  /***
   *Leaky linear rectifier unit
   *@return 0.01x for x < 1 else x
   ***/
  lReLU: function(wa)
  {
      if(wa < 1)
      {
          return 0.01*wa;
      }
      return wa;
  },
  dlReLU: function(x)
  {
      if(x > 0)
      {
          return 1;
      }
      
      return 0.01;
  }
};

window.nn.util.new2dArray = function(width, height)
{
    var array = [];
    
    for(i = 0; i < width; i++)
    {
        array[i] = [];
        for(j = 0; j < height; j++)
        {
            array[i][j] = 0;
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
            matrix[0] = window.nn.util.new2dArray(inputs, width);
        }
        else if(layer < layers)
        {
            matrix[layer] = window.nn.util.new2dArray(width, width);
        }
        else
        {
            matrix[layers] = window.nn.util.new2dArray(width, outputs);
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
*@param [Function] ϕ The activation function to use
***/
window.nn.NNet = function(inputs, layers, width, outputs, ϕ, dϕ)
{
    this.layers = layers;
    this.inputs = inputs;
    this.width = width;
    this.outputs = outputs;
    
    this.ϕ = ϕ || activationFunctions.identity;
    this.dϕ = dϕ || function(){return 1;};
    
    this.weights = window.nn.util.newWeightMatrix(layers, width, inputs, outputs);
    this.values = window.nn.util.newValueMatrix(layers, width, inputs, outputs);
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
    }
    else
    {
        throw "Input number mismatch!";
    }
};

window.nn.NNet.prototype.getInputs = function()
{
    return this.values[0];
};

window.nn.NNet.prototype.getOutputs = function()
{
    return this.values[this.layers];
};

window.nn.NNet.prototype.processLayer(layer, inputs, outputs)
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
        
        this.setValue(layer+1, n2, this.ϕ(wa));
    }
};

window.nn.NNet.prototype.setLossFunction = function(L, dL)
{
    this.L = L;
};

window.nn.NNet.prototype.L = function(expected, output)
{
    return Math.pow(expected - output, 2);
};

window.nn.NNet.prototype.dL = function(expected, output)
{
    return 2 * (expected - output);
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

/***
 * 
 *@param dataset an array of object of form {inputs: [1, 2, etc...], outputs: [0, 1, 0, etc...]}
 ***/
window.nn.NNet.prototype.train = function(dataset)
{
    
}
