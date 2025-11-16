using Micrograd.Console.Extensions;
using Micrograd.NN;
using Spectre.Console;
namespace Micrograd.Console.Examples;

/// <summary>
/// Example of the image recognition using MLP based on the <see cref="Value"/> object.
/// The example uses MNIST dataset. Note that this is just a example and training 
/// on the big amount of images on CPU using <see cref="Value"/> object is very slow.
/// </summary>
public partial class Examples
{
    private const int ImageBatchSize = 40;

    private const string trainImagesPath = @"<path-to-trainining-images\train-images.idx3-ubyte>";
    private const string trainLabelsPath = @"<path-to-trainining-labels\train-labels.idx1-ubyte>";

    private const string testImagesPath = @"<path-to-test-images\t10k-images.idx3-ubyte>";
    private const string testLabelsPath = @"<path-to-test-images\t10k-labels.idx1-ubyte>";

    private record class IncorrectImage(int Index, int Expected, int Predicted);

    public static void RunImageRecognition()
    {
        int iterationsCount = AnsiConsole.Ask<int>("[yellow]Number of training iterations ?[/]");
        int imagesToTrainCount = AnsiConsole.Ask<int>("[yellow]Number of images to train ?[/]");
        int imagesToTestCount = AnsiConsole.Ask<int>("[yellow]Number of images to test ?[/]");

        // Define MLP with 784 inputs, 2 hidden layers with 28 and 16 neurons and 10 outputs
        // The 784 inputs are the pixels of the image (28x28).
        // The 10 outputs are the probabilities of the image to be one of the 10 digits.
        MLP mlp = new(28 * 28, [28, 16, 10]);

        TrainModel(mlp, iterationsCount, imagesToTrainCount);
        TestModel(mlp, imagesToTestCount);
    }

    private static void TestModel(MLP mlp, int imagesToTestCount)
    {
        (double[][]? testImages, int[]? testLabels) = ReadImages(testImagesPath, testLabelsPath, imagesToTestCount);
        int correct = 0;
        HashSet<IncorrectImage> incorrectImages = [];
        for (int i = 0; i < testImages.Length; i++)
        {
            Value[] values = Array.ConvertAll<double, Value>(testImages[i], x => x);
            Value[] preds = mlp.Forward(values);
            Value? max = preds.Max();
            int maxIndex = Array.IndexOf(preds, max);
            if (maxIndex == testLabels[i])
                correct++;
            else
                incorrectImages.Add(new IncorrectImage(i, maxIndex, testLabels[i]));
        }

        AnsiConsole.Write(new BreakdownChart()
            .Width(60)
            .AddItem("Correct", correct, Color.Green)
            .AddItem("Incorrect", imagesToTestCount - correct, Color.Red));

        AnsiConsole.MarkupLine($"Accuracy: [yellow]{(float)correct / testImages.Length * 100.0:0.###}%[/]");

        // Manual prediction mode
        while (AnsiConsole.Confirm("Select image to predict ?"))
        {
            int index = AnsiConsole.Ask<int>("[yellow] Enter image index ?[/]");
            Value[] values = Array.ConvertAll<double, Value>(testImages[index], x => x);
            Value[] preds = mlp.Forward(values);
            Value? max = NewMethod(preds);
            int maxIndex = Array.IndexOf(preds, max);
            AnsiConsole.MarkupLine($"Predicted: [yellow]{maxIndex}[/]");
            AnsiConsole.MarkupLine($"Expected: [yellow]{testLabels[index]}[/]");

            double dimension = Math.Pow(testImages[index].Length, 0.5);

            PrintImage((int)dimension, (int)dimension, testImages[index]);
        }
    }

    private static Value? NewMethod(Value[] preds) => preds.Max();

    private static void TrainModel(MLP mlp, int iterationsCount, int imagesToTrainCount, double learningRate = 0.05)
    {
        (double[][]? trainImages, int[]? trainLabels) = ReadImages(trainImagesPath, trainLabelsPath, imagesToTrainCount);
        for (int it = 0; it < iterationsCount; it++)
        {
            for (int j = 0; j < trainImages.Length; j += ImageBatchSize)
            {
                Value totalLoss = new(0.0);
                int imagesToTake = j + ImageBatchSize > trainImages.Length ? trainImages.Length - j : ImageBatchSize;

                double[][] batchImages = trainImages.Skip(j).Take(imagesToTake).ToArray();
                int[] batchLabels = trainLabels.Skip(j).Take(imagesToTake).ToArray();

                for (int k = 0; k < batchLabels.Length; k++)
                {
                    Value[] values = Array.ConvertAll<double, Value>(batchImages[k], x => x);
                    Value[] preds = mlp.Forward(values);
                    double[] target = new double[10];
                    for (int i = 0; i < target.Length; i++)
                        target[i] = i == batchLabels[k] ? 1.0 : -1.0;

                    // Calculate the loss
                    Value imageLoss = new(0);
                    for (int i = 0; i < preds.Length; i++)
                        imageLoss += (preds[i] - target[i]).Pow(2);

                    imageLoss /= preds.Length;
                    totalLoss += imageLoss;
                }

                mlp.ZeroGrad();
                totalLoss.Backward();

                foreach (Value p in mlp.GetParameters())
                    p.Data -= learningRate * p.Gradient;

                AnsiConsole.MarkupLine($"Loss: [yellow]{totalLoss.Data}[/]");
            }
        }
    }

    private static void PrintImage(int rowsCount, int columnsCount, double[] bytes)
    {
        Canvas canvas = new(rowsCount, columnsCount);
        for (int i = 0; i < rowsCount; i++)
        {
            for (int j = 0; j < columnsCount; j++)
            {
                byte value = (byte)(bytes[i * rowsCount + j] * 255.0);
                canvas.SetPixel(j, i, value == 0 ? Color.Yellow1 : new Color(value, value, value));
            }
        }

        AnsiConsole.Write(canvas);
    }

    private static (double[][] images, int[] labels) ReadImages(string imagesPath, string labelsPath, int imagesToRead)
    {
        using BinaryReader imageReader = new(File.OpenRead(imagesPath));
        using BinaryReader labelReader = new(File.OpenRead(labelsPath));

        int magicNumber = imageReader.ReadInt32Endian();
        if (magicNumber != 2051)
            throw new Exception($"Invalid magic number {magicNumber}!");

        int labelsMagicNumber = labelReader.ReadInt32Endian();
        if (labelsMagicNumber != 2049)
            throw new Exception("Invalid label file magic number.");

        int imagesCount = imageReader.ReadInt32Endian();
        int labelsCount = labelReader.ReadInt32Endian();
        if (imagesCount != labelsCount)
            throw new Exception("Image count and label count do not match.");

        if (imagesToRead > imagesCount)
            throw new Exception("Images to read is greater than images count.");

        int rowsCount = imageReader.ReadInt32Endian();
        int columnsCount = imageReader.ReadInt32Endian();
        int imageSize = rowsCount * columnsCount;

        double[][] images = new double[imagesToRead][];
        int[] labels = new int[imagesToRead];
        for (int i = 0; i < imagesToRead; i++)
        {
            images[i] = new double[imageSize];
            for (int j = 0; j < images[i].Length; j++)
                images[i][j] = imageReader.ReadByte() / 255.0;
            labels[i] = labelReader.ReadByte();
        }

        return (images, labels);
    }
}
