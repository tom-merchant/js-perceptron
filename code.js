

function main()
{
  var dataset = [
    {inputs: [1, 2], outputs: [3]},
    {inputs: [1.5, 0.5], outputs: [2]},
    {inputs: [0.1, 0.2], outputs: [0.3]},
    {inputs: [1, 1], outputs: [2]}
  ];
  var xorNet = new nn.NNet(2, 1, 2, 1, nn.activationFunctions.tanh, nn.activationFunctions.dtanh, 0.1, true);

  for(var i = 0; i < 1000; i++)
  {
    xorNet.train(dataset);
  }

  xorNet.calculateLoss(dataset);
}
