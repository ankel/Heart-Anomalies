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
        static double MEstimate = 0.5;

        static int Clasification(double[,] learnedData, int[] currentCase)
        {
            double  p0 = Math.Log(learnedData[0,0]),
                    p1 = Math.Log(learnedData[1,0]);

            for (int i = 1; i < (learnedData.Length / learnedData.Rank); ++i)
            {
                if (currentCase[i] == 1)
                {
                    p0 += Math.Log(learnedData[0, i]);
                    p1 += Math.Log(learnedData[1, i]);
                }
            }
            
            return p1 > p0 ? 1 : 0;
        }

        static double[,] Learn(string learningData)
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

            double[,] learned = new double[2, featuresCount];
            for (int i = 0; i < featuresCount; ++i)
            {
                learned[0, i] = (counts[0, i] + MEstimate) / (counts[2, i] + MEstimate);
                learned[1, i] = (counts[1, i] + MEstimate) / (counts[2, i] + MEstimate);
            }

            return learned;
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
            double[,] origData = Learn(data["orig"].Item1);
            int featureCount = origData.Length / origData.Rank;
            for (int i = 0; i < origData.Rank; ++i)
            {
                for (int j = 0; j < featureCount; ++j)
                {
                    Console.Write(origData[i,j].ToString("F3") + ",");
                }
                Console.WriteLine();
            }
            
            Console.ReadLine();
        }
    }
}
