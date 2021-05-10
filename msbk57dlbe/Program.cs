using System;
using System.Collections.Generic;
using System.Linq;
using CNTK;
using System.IO;
using System.Globalization;

namespace msbk57dlbe
{
    class msbk57dlbe
    {
        /// <summary>
        /// asd
        /// </summary>
        readonly int[] layers = new int[] { DataSet.InputSize, 3, 4, 5, DataSet.OutputSize };
        //asd
        const int batchSize = 15;
        const int epochCount = 1000;

        readonly Variable x;
        readonly Function y;

        public msbk57dlbe()
        {
            x = Variable.InputVariable(new int[] { layers[0] }, DataType.Float, "x");

            Function lastLayer = x;
            for (int i = 0; i < layers.Length - 1; i++)
            {
                Parameter weight = new Parameter(new int[] { layers[i + 1], layers[i] }, DataType.Float, CNTKLib.GlorotNormalInitializer());
                Parameter bias = new Parameter(new int[] { layers[i + 1] }, DataType.Float, CNTKLib.GlorotNormalInitializer());

                Function times = CNTKLib.Times(weight, lastLayer);
                Function plus = CNTKLib.Plus(times, bias);
                lastLayer = CNTKLib.Sigmoid(plus);
            }

            y = lastLayer;
        }

        public msbk57dlbe(string filename)
        {
            y = Function.Load(filename, DeviceDescriptor.CPUDevice);
            x = y.Arguments.First(x => x.Name == "x");
        }

        public void Train(DataSet ds)
        {
            Variable yt = Variable.InputVariable(new int[] { DataSet.OutputSize }, DataType.Float);
            Function loss = CNTKLib.BinaryCrossEntropy(y, yt);

            Function y_rounded = CNTKLib.Round(y);
            Function y_yt_equal = CNTKLib.Equal(y_rounded, yt);

            Learner learner = CNTKLib.SGDLearner(new ParameterVector(y.Parameters().ToArray()), new TrainingParameterScheduleDouble(1.0, batchSize));
            Trainer trainer = Trainer.CreateTrainer(y, loss, y_yt_equal, new List<Learner>() { learner });

            for (int epochI = 0; epochI <= epochCount; epochI++)
            {
                double sumLoss = 0;
                double sumEval = 0;

                for (int batchI = 0; batchI < ds.Count / batchSize; batchI++)
                {
                    Value x_value = Value.CreateBatch(x.Shape, ds.Input.GetRange(batchI * batchSize * DataSet.InputSize, batchSize * DataSet.InputSize), DeviceDescriptor.CPUDevice);
                    Value yt_value = Value.CreateBatch(yt.Shape, ds.Output.GetRange(batchI * batchSize * DataSet.OutputSize, batchSize * DataSet.OutputSize), DeviceDescriptor.CPUDevice);
                    var inputDataMap = new Dictionary<Variable, Value>()
                    {
                        { x, x_value },
                        { yt, yt_value }
                    };

                    trainer.TrainMinibatch(inputDataMap, false, DeviceDescriptor.CPUDevice);
                    sumLoss += trainer.PreviousMinibatchLossAverage() * trainer.PreviousMinibatchSampleCount();
                    sumEval += trainer.PreviousMinibatchEvaluationAverage() * trainer.PreviousMinibatchSampleCount();
                }
                Console.WriteLine(String.Format("{0}\tloss:{1}\teval:{2}", epochI, sumLoss / ds.Count, sumEval / ds.Count));
            }
        }

        public float Prediction(List<float> testpred)
        {
            var inputDataMap = new Dictionary<Variable, Value>() { { x, Value.CreateBatch(x.Shape, DataSet.NormalizePred(testpred), DeviceDescriptor.CPUDevice) } };
            var outputDataMap = new Dictionary<Variable, Value>() { { y, null } };
            y.Evaluate(inputDataMap, outputDataMap, DeviceDescriptor.CPUDevice);
            return outputDataMap[y].GetDenseData<float>(y)[0][0];
        }

        Value LoadInput(List<float> testpred)
        {
            List<float> x_store = DataSet.Normalize(testpred);
            return Value.CreateBatch(x.Shape, DataSet.Normalize(testpred), DeviceDescriptor.CPUDevice);
        }

        public void Save(string filename)
        {
            y.Save(filename);
        }

        public double Evaluate(DataSet ds)
        {
            Variable yt = Variable.InputVariable(new int[] { DataSet.OutputSize }, DataType.Float);

            Function y_rounded = CNTKLib.Round(y);
            Function y_yt_equal = CNTKLib.Equal(y_rounded, yt);
            Evaluator evaluator = CNTKLib.CreateEvaluator(y_yt_equal);

            double sumEval = 0;
            for (int batchI = 0; batchI < ds.Count / batchSize; batchI++)
            {
                Value x_value = Value.CreateBatch(x.Shape, ds.Input.GetRange(batchI * batchSize * DataSet.InputSize, batchSize * DataSet.InputSize), DeviceDescriptor.CPUDevice);
                Value yt_value = Value.CreateBatch(yt.Shape, ds.Output.GetRange(batchI * batchSize * DataSet.OutputSize, batchSize * DataSet.OutputSize), DeviceDescriptor.CPUDevice);
                var inputDataMap = new UnorderedMapVariableValuePtr()
                    {
                        { x, x_value },
                        { yt, yt_value }
                    };

                sumEval += evaluator.TestMinibatch(inputDataMap, DeviceDescriptor.CPUDevice) * batchSize;
            }
            return sumEval / ds.Count;
        }
    }

    public class DataSet
    {
        public const int InputSize = 7;
        public List<float> Input { get; set; } = new List<float>();

        public const int OutputSize = 1;
        public List<float> Output { get; set; } = new List<float>();

        public int Count { get; set; }
        public static int Counter { get; set; }

        //public static int BinaryStartingIndex = 0;

        //public static int BinaryEndIndex = 0;



        public DataSet(string filename)
        {
            LoadData(filename);
        }

        void LoadData(string filename)
        {
            Count = 0;
            foreach (String line in File.ReadAllLines(filename))
            {
                var floats = Normalize(line.Split(';').Select(x => float.Parse(x, CultureInfo.InvariantCulture.NumberFormat)).ToList());
                Input.AddRange(floats.GetRange(0, InputSize));
                Output.Add(floats[InputSize]);
                Count++;
            }
        }

        static float[] minValues;
        static float[] maxValues;
        static float[] true_percents;

        public static List<float> Normalize(List<float> floats)
        {
            List<float> normalized = new List<float>();
            for (int i = 0; i < floats.Count; i++)
            {
                normalized.Add((floats[i] - minValues[i]) / (maxValues[i] - minValues[i]));
            }
            return normalized;
        }

        public static List<float> NormalizePred(List<float> floats)
        {
            List<float> normalized = new List<float>();
            for (int i = 0; i < floats.Count - 1; i++)
            {
                normalized.Add((floats[i] - minValues[i]) / (maxValues[i] - minValues[i]));
            }
            return normalized;
        }

        public static void LoadMinMax(string filename)
        {
            foreach (String line in File.ReadAllLines(filename))
            {
                var floats = line.Split(';').Select(x => float.Parse(x, CultureInfo.InvariantCulture.NumberFormat)).ToList();
                if (minValues == null)
                {
                    minValues = floats.ToArray();
                    maxValues = floats.ToArray();
                }
                else
                {
                    for (int i = 0; i < floats.Count; i++)
                    {
                        if (floats[i] < minValues[i])
                        {
                            minValues[i] = floats[i];
                        }
                        else
                        {
                            if (floats[i] > maxValues[i])
                            {
                                maxValues[i] = floats[i];
                            }
                        }
                    }

                }
            }
        }

        //public static List<float> NormalizeWithTruePercent(List<float> floats)
        //{
        //    List<float> normalized = new List<float>();
        //    for (int i = 0; i < floats.Count; i++)
        //    {
        //        if (i < DataSet.BinaryStartingIndex)
        //        {
        //            normalized.Add((floats[i] - minValues[i]) / (maxValues[i] - minValues[i]));
        //        }
        //        else if (i >= DataSet.BinaryStartingIndex && i < floats.Count)
        //        {
        //            if (floats[i] == 1)
        //            {
        //                normalized.Add(true_percents[i - DataSet.BinaryStartingIndex]);
        //            }
        //            else
        //            {
        //                normalized.Add(1 - true_percents[i - DataSet.BinaryStartingIndex]);
        //            }

        //        }
        //        else
        //        {
        //            normalized.Add(floats[i]);
        //        }
        //    }

        //    return normalized;
        //}

        //public static List<float> NormalizePredWithTruePercent(List<float> floats)
        //{
        //    List<float> normalized = new List<float>();
        //    for (int i = 0; i < floats.Count - 1; i++)
        //    {
        //        if (i < 7)
        //        {
        //            normalized.Add((floats[i] - minValues[i]) / (maxValues[i] - minValues[i]));
        //        }
        //        else if (i >= DataSet.BinaryStartingIndex && i < floats.Count)
        //        {
        //            if (floats[i] == 1)
        //            {
        //                normalized.Add(true_percents[i - DataSet.BinaryStartingIndex]);
        //            }
        //            else
        //            {
        //                normalized.Add(1 - true_percents[i - DataSet.BinaryStartingIndex]);
        //            }
        //        }
        //    }
        //    return normalized;
        //}

        //public static void LoadMinMaxWithTruePurcent(string filename)
        //{
        //    foreach (String line in File.ReadAllLines(filename))
        //    {
        //        var floats = line.Split(';').Select(x => float.Parse(x, CultureInfo.InvariantCulture.NumberFormat)).ToList();
        //        if (minValues == null)
        //        {
        //            minValues = new float[DataSet.BinaryStartingIndex];
        //            for (int i = 0; i < DataSet.BinaryStartingIndex; i++)
        //            {
        //                minValues[i] = floats[i];
        //            }
        //            maxValues = new float[DataSet.BinaryStartingIndex];
        //            for (int i = 0; i < DataSet.BinaryStartingIndex; i++)
        //            {
        //                maxValues[i] = floats[i];
        //            }
        //        }
        //        for (int i = 0; i < DataSet.BinaryStartingIndex; i++)
        //        {
        //            if (floats[i] < minValues[i])
        //            {
        //                minValues[i] = floats[i];
        //            }
        //            else
        //            {
        //                if (floats[i] > maxValues[i])
        //                {
        //                    maxValues[i] = floats[i];
        //                }
        //            }
        //        }
        //    }
        //}

        //public static float[] GetTruePercents(string filename)
        //{
        //    true_percents = new float[DataSet.BinaryEndIndex];
        //    for (int i = 0; i < DataSet.BinaryEndIndex - DataSet.BinaryStartingIndex; i++)
        //    {
        //        true_percents[i] = 0;
        //    }
        //    foreach (String line in File.ReadAllLines(filename))
        //    {
        //        var floats = line.Split(';').Select(x => float.Parse(x, CultureInfo.InvariantCulture.NumberFormat)).ToList();
        //        for (int i = DataSet.BinaryStartingIndex; i < InputSize - 1; i++)
        //        {
        //            if (floats[i] == 1)
        //            {
        //                true_percents[i - DataSet.BinaryStartingIndex]++;
        //            }
        //        }
        //        Counter++;
        //    }
        //    for (int i = 0; i < true_percents.Length; i++)
        //    {
        //        true_percents[i] = true_percents[i] / Counter;
        //    }
        //    return true_percents;
        //}
    }

    public class Program
    {
        void FileTest(msbk57dlbe app)
        {
            string[] testData = File.ReadAllLines(@"tests.txt");
            int TP = 0, TN = 0, FP = 0, FN = 0;
            int goodPrediction = 0, wrongPrediction = 0;
            foreach (string line in testData)
            {
                List<float> values = line.Split(';').Select(x => float.Parse(x, CultureInfo.InvariantCulture.NumberFormat)).ToList();
                float pred = app.Prediction(values);

                if (Math.Round(pred) != values[values.Count-1])
                {
                    Console.WriteLine("--- " + pred + "\t" + line);
                    wrongPrediction++;
                    if (pred == 1)
                    {
                        FP++;
                    }
                    else
                    {
                        FN++;
                    }

                }
                else
                {
                    Console.WriteLine("+++ " + pred + "\t" + line);
                    goodPrediction++;
                    if (Math.Round(pred) == 1)
                    {
                        TP++;
                    }
                    else
                    {
                        TN++;
                    }
                }
            }
            float accuracy = (float)(TP + TN) / (TP + FP + TN + FN);
            float precision = (float)TP / (TP + FP);
            float sensitivity = (float)TP / (TP + FN);
            float F1 = 2 * (precision * sensitivity) / (precision + sensitivity);
            Console.WriteLine(String.Format("True positive:\t{0}\nTrue negative:\t{1}\nFalse positive:\t{2}\nFalse negative:\t{3}", TP, TN, FP, FN));
            Console.WriteLine(String.Format("Accuracy:\t{0}\nPrecision:\t{1}\nSensitivity:\t{2}\nF1 score:\t{3}", accuracy, precision, sensitivity, F1));
            Console.WriteLine(String.Format("Good prediction:{0} ({1}%)", goodPrediction, 100f * goodPrediction / testData.Count()));
            Console.WriteLine(String.Format("Wrong prediction:{0} ({1}%)", wrongPrediction, 100f * wrongPrediction / testData.Count()));
        }
        static void Main(string[] args)
        {
            DataSet.LoadMinMax(@"trains.txt");
            //DataSet.GetTruePercents(@"trains.txt", DataSet.BinaryStartingIndex);
            DataSet trainDS = new DataSet(@"trains.txt");
            DataSet testDS = new DataSet(@"tests.txt");

            msbk57dlbe app = new msbk57dlbe();
            app.Train(trainDS);
            app.Save(@"DeepNetworkTest2.model");
            //msbk57dlbe app = new msbk57dlbe(@"DeepNetworkTest.model"); //vagy DeepNetwork.model
            Console.WriteLine("Eval train:" + app.Evaluate(trainDS));
            Console.WriteLine("Eval test:" + app.Evaluate(testDS));
            new Program().FileTest(app);
            Console.ReadKey();
        }
    }
}
