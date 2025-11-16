namespace Micrograd.Utilities;

internal class Randomizer
{
    private static readonly Lock syncLock = new();
    private static readonly Random random = new();

    public static double GetRandomValue()
    {
        lock (syncLock)
        {
            return random.NextDouble() * 2 - 1;
        }
    }
}
