using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Heart_Anomalies
{
    class Program
    {
        static Dictionary<string, Tuple<string, string>> data = new Dictionary<string, Tuple<string, string>>()
        {
            {"itg", new Tuple<string, string>(@"..\..\..\heart-data\spect-itg.train.csv", @"..\..\..\heart-data\spect-itg.test.csv")},
            {"orig", new Tuple<string, string>(@"..\..\..\heart-data\spect-orig.train.csv", @"..\..\..\heart-data\spect-orig.test.csv")},
            {"resplit", new Tuple<string, string>(@"..\..\..\heart-data\spect-resplit.train.csv", @"..\..\..\heart-data\spect-resplit.test.csv")},
        };
        static double MConst = 10 * double.Epsilon;
        static bool debug = false;
        static System.IO.StreamWriter debugFile = new System.IO.StreamWriter(@"..\..\..\debug.txt", false);

        /// <summary>
        /// Given a count result, check the current case for a match
        /// </summary>
        /// <param name="learnedData">a [3,x] int array contains counts of each feature</param>
        /// <param name="currentCase">a [x] int array contains current case's features</param>
        /// <returns>0 if abnormal, 1 if normal</returns>
        static int Clasification(int[,] learnedData, int[] currentCase)
        {
            double  p0 = Math.Log(MEstimate(learnedData[0,0], learnedData[2,0], MConst)),
                    p1 = Math.Log(MEstimate(learnedData[1,0], learnedData[2,0], MConst));

            int length = learnedData.Length / (learnedData.Rank + 1);
            for (int i = 1; i < length; ++i)
            {
                if (currentCase[i] == 1)
                {
                    p0 += Math.Log(MEstimate(learnedData[0, i], learnedData[0, 0], MConst));
                    p1 += Math.Log(MEstimate(learnedData[1, i], learnedData[1, 0], MConst));
                }
                else
                {
                    p0 += Math.Log(MEstimate(learnedData[0, 0] - learnedData[0, i], learnedData[0, 0], MConst));
                    p1 += Math.Log(MEstimate(learnedData[1, 0] - learnedData[1, i], learnedData[1, 0], MConst));
                }
            }
            
            return p1 > p0 ? 1 : 0;
        }

        /// <summary>
        /// Counting function
        /// </summary>
        /// <param name="learningData">input csv file</param>
        /// <returns>a [3,x] int array contains all counts</returns>
        static int[,] Learn(string learningData)
        {
            int [,] counts;
            string[] line;
            int[] fields;
            int featuresCount;
            using (System.IO.StreamReader inFile = new System.IO.StreamReader(learningData))
            {
                line = inFile.ReadLine().Split(',');        // read first line to determine the number of features
                featuresCount = line.Length;                // number of features
                counts = new int[3, featuresCount];
                fields = new int[featuresCount];

                inFile.BaseStream.Seek(0, System.IO.SeekOrigin.Begin);      // reset stream
                inFile.DiscardBufferedData();

                while (!inFile.EndOfStream)
                {
                    line = inFile.ReadLine().Split(',');
                    ParseFields(line, fields);
                    ++counts[fields[0], 0];         // increase the respective counts
                    ++counts[2, 0];                 //increase the total counts
                    for (int i = 1; i < featuresCount; ++i)
                    {
                        counts[fields[0], i] += fields[i];  // increase the respective counts
                        counts[2, i] += fields[i];          // increase the total counts
                    }
                }
            }

            //double[,] learned = new double[2, featuresCount];
            //learned[0, 0] = MEstimate(counts[0, 0], counts[2, 0]);
            //learned[1, 0] = MEstimate(counts[1, 0], counts[2, 0]);
            //for (int i = 1; i < featuresCount; ++i)
            //{
            //    learned[0, i] = MEstimate(counts[0, i], counts[0, 0]);
            //    learned[1, i] = MEstimate(counts[1, i], counts[1, 0]);
            //}

            if (debug)
            {
                PrintArr(counts, debugFile);
                //PrintArr(learned, debugFile);
            }

            return counts;
        }

        /// <summary>
        /// M-estimation of a quotient
        /// </summary>
        /// <param name="p">numerator</param>
        /// <param name="p_2">denominator</param>
        /// <param name="MConst">M constant</param>
        /// <returns>the resulting quotient</returns>
        private static double MEstimate(int p, int p_2, double MConst)
        {
            return (p + MConst) / (p_2 + MConst);
        }

        private static void PrintArr(Array arr, System.IO.StreamWriter output)
        {
            int length = arr.Length / (arr.Rank + 1);
           
            for (int i = 0; i < arr.Rank; ++i)
            {
                for (int j = 0; j < length; ++j)
                {
                    output.Write(arr.GetValue(i,j) + ",");
                }
                output.WriteLine();
            }
        }

        private static void ParseFields(string[] line, int[] fields)
        {
            for (int i = 0; i < line.Length; ++i)
            {
                fields[i] = Convert.ToInt32(line[i]);
            }
        }

        static void Main(string[] args)
        {
            foreach(var d in data)
            {
                int[,] learned = Learn(d.Value.Item2);
                int[] actual = new int[2];
                int[] predic = new int[2];
                using (System.IO.StreamReader inFile = new System.IO.StreamReader(d.Value.Item1))
                {
                    
                    while (!inFile.EndOfStream)
                    {
                        string[] line = inFile.ReadLine().Split(',');
                        int[] fields = new int[line.Length];
                        ParseFields(line, fields);
                        ++actual[fields[0]];
                        int n = Clasification(learned, fields);
                        if (n == fields[0])
                        {
                            ++predic[n];
                        }
                    }
                }
                if (debug)
                {
                    //debugFile.Close();
                    //break;
                }

                Console.WriteLine(String.Format("{0} {1} {2} {3}", d.Key,
                                                                    AccuRate(predic.Sum(), actual.Sum()),
                                                                    AccuRate(predic[0], actual[0]),
                                                                    AccuRate(predic[1], actual[1])));
            }
            debugFile.Close();
            Console.ReadLine();
        }

        private static string AccuRate(int p, int p_2)
        {
            return String.Format("{0}/{1}({2:F2})", p, p_2, (double)p / p_2);
        }
    }
}
