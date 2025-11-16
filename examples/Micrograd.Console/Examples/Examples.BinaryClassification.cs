using Micrograd.NN;
using ScottPlot;
using Spectre.Console;
namespace Micrograd.Console.Examples;

public partial class Examples
{
    /// <summary>
    /// Binary classification example.
    /// </summary>
    public static void RunBinaryClassificationExample()
    {
        // Define learning parameters : 3 inputs, 2 hidden layers with 4 neurons and single output
        MLP mlp = new(3, [4, 4, 1]);

        // Create training dataset
        Value[][] matrix =
        [
            [2.0, 3.0, -1.0],
            [3.0, -1.0, 0.5],
            [0.5, 1.0, 1.0],
            [1.0, 1.0, -1.0]
        ];

        // Desired targets
        double[] ys = [1.0, -1.0, -1.0, 1.0];

        int numberOfIteration = AnsiConsole.Ask<int>("[yellow]Number of iterations ?[/]");
        double[] losses = new double[numberOfIteration];

        // Train the network anc calculate the loss function
        for (int it = 0; it < numberOfIteration; it++)
        {
            Value loss = new(0.0);
            for (int j = 0; j < ys.Length; j++)
            {
                Value pred = mlp.Forward(matrix[j])[0];
                loss += (pred - ys[j]).Pow(2);
            }

            // Reset gradient
            mlp.ZeroGrad();

            // Backpropagation
            loss.Backward();

            // Update the parameters (weights and biases)
            foreach (Value p in mlp.GetParameters())
                p.Data -= 0.01 * p.Gradient;

            losses[it] = loss.Data;

            AnsiConsole.WriteLine($"Loss: {loss.Data}");
        }

        // Draw the loss function
        if (AnsiConsole.Confirm("Draw the loss function ?"))
        {
            Plot plot = new();

            plot.XLabel("Iteration");
            plot.YLabel("Loss");

            double[] xValues = [.. Enumerable.Range(0, numberOfIteration).Select(v => (double)v)];
            plot.Add.Scatter(xValues, losses);

            plot.GetImage(800, 600).SavePng("loss.png");
            AnsiConsole.MarkupLine($"[green]Loss function saved.[/]");
        }
    }
}
