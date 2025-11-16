using Micrograd.Console.Extensions;
namespace Micrograd.Console.Examples;

public partial class Examples
{
    /// <summary>
    /// Calculation of the gradient for the 'y = x1w1 + x2w2 + b' expression.
    /// </summary>
    public static void RunFunctionGradient()
    {
        Value x1 = new(2.0) { Label = "x1" };
        Value x2 = new(0.0) { Label = "x2" };
        Value w1 = new(-3.0) { Label = "w1" };
        Value w2 = new(1.0) { Label = "w2" };

        Value b = new(6.88137358) { Label = "b" };

        Value x1w1 = x1 * w1; x1w1.Label = "x1w1";
        Value x2w2 = x2 * w2; x2w2.Label = "x2w2";

        Value x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.Label = "x1w1x2w2";
        Value n = x1w1x2w2 + b; n.Label = "n";
        Value o = n.Tanh(); o.Label = "o";

        o.Backward();
        o.PrintAsTree();
    }
}
