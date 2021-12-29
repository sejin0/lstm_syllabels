package com.cyd.lstmnextsyllable;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.core.util.Pair;
import androidx.recyclerview.widget.SortedList;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SortedMap;
import java.util.TreeMap;

public class PredictModel {
    public static final String LOG_TAG = PredictModel.class.getSimpleName();

    int topK;
    Interpreter interpreter;
    Map<String, Integer> s2i = null;
    Map<Integer, String> i2s = new HashMap<>();

//    static {
//        MainActivity activity = null;
//        PredictModel model = new PredictModel(activity, 10, "lstmmodel.tflite", "dict.csv");
//
//
//        List<Pair<String, Float>> predictions = model.predict("우리는");
//        for(Pair<String, Float> e : predictions) {
//            Log.d("PredictionModel", e.first + "=>" + e.second);
//        }
//
//
//    }

    public PredictModel(Context context, int topK, String modelPath, String dictPath) throws Exception {
          interpreter = createInterpreter(context, modelPath);
          this.topK = topK;
          s2i = createDictionary(context, dictPath);
          for(Map.Entry<String, Integer> entry : s2i.entrySet()) {
              i2s.put(entry.getValue(), entry.getKey());
          }
    }

    private Interpreter createInterpreter(Context context, String path) throws Exception {
        FileChannel fileChannel = null;
        try {
            AssetFileDescriptor fileDescriptor = context.getAssets().openFd(path);
            Log.i(LOG_TAG, "WE GOT FILE DESCRIPTOR");
            fileChannel = inputStreamFromAsset(context, path).getChannel();
            Log.i(LOG_TAG, "FILE CHANNEL OPENED");
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            MappedByteBuffer buffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
            Log.i(LOG_TAG, "BYTE BUFFER MAPPED");

            Interpreter result = new Interpreter(buffer);
            Log.i(LOG_TAG, "INTERPRETER CREATED");

            return result;
        } finally {
//            try {
//                Log.i(LOG_TAG, "CLOSING FILE CHANNEL");
//                if (fileChannel != null) fileChannel.close();
//            } catch (Exception e) {
//                e.printStackTrace();
//            }
        }
    }

    private FileInputStream inputStreamFromAsset(Context context, String path) throws IOException, IOException {
        AssetFileDescriptor fileDescriptor = context.getAssets().openFd(path);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        return inputStream;
    }

    private Map<String, Integer> createDictionary(Context context, String path) throws IOException, IOException {
        Map<String, Integer> result = new HashMap<>();
        BufferedReader reader = null;
        try {
            Log.i(LOG_TAG, "READING DICTIONARY");
            reader = new BufferedReader(new InputStreamReader(context.getAssets().open(path)));
            Log.i(LOG_TAG, "READER CREATED");
            String line = reader.readLine();
            int c = 0;
            String s[] = null;
            try {
                while (line != null) {
                    c++;
                    s = line.split("\\|");
                    if ((c > 925 && c < 1000) || c < 10)
                        Log.i(LOG_TAG, "READ: " + line + " " + s[0] + ">" + s[1]);
                    if (s.length >= 2) {
                        int index = Integer.parseInt(s[1]);
                        result.put(s[0], index);
                    }
                    line = reader.readLine();
                }
            } catch(Exception e) {
                Log.e(LOG_TAG, "EXCEPTION: " + e.getClass().getSimpleName() + " " + e.getMessage());
                if (s != null) {
                    Log.e(LOG_TAG, "line " + line);
                    Log.e(LOG_TAG, "s[0] " + s[0]);
                    Log.e(LOG_TAG, "s[1] " + s[1]);
                    Log.e(LOG_TAG, "s[3] " + s[1]);
                }
            }
            Log.i(LOG_TAG, "DICTIONARY SIZE=" + result.size());
            return result;
        } finally {
//            try {
//                if (reader != null)
//                    try {
//                        reader.close();
//                    } catch(Exception e) {
//                        e.printStackTrace();
//                    }
//            } catch (Exception e) {
//            }
        }
    }

    // one-hot encoding
    private float[][][] encode(String input) {
        float[][][] result = new float[1][input.length()][s2i.size()];
        for(int i = 0;i < result.length;i++) {
            for (int j = 0; j < result[i].length; j++) {
                for(int k = 0; k < result[i][j].length;k++) {
                    result[i][j][k] = 0.0f;
                }
            }
        }

        for(int i = 0;i < 3;i++) {
            if (i < input.length()) {
                String syllable = String.valueOf(input.charAt(i));
                Integer index = s2i.get(syllable);
                if (index != null) {
                    result[0][i][index] = 1.0f;
                } else { // set OOV
                }
            }
        }

        return result;
    }

    public List<Pair<String, Float>> predict(String input) {
        List<Pair<String, Float>> result = new ArrayList<>(topK);

        //Tensor inputTensor = interpreter.getInputTensor(0);
        //Tensor outputTensor = interpreter.getOutputTensor(0);

        float[][][] encodedInput = encode(input);
        float[][] prediction = new float[1][encodedInput[0][0].length];

        interpreter.run(encodedInput, prediction);

        List<Pair<String, Float>> list= new ArrayList<Pair<String, Float>>();
        for(int i = 0;i < prediction[0].length;i++) {
            float p = prediction[0][i];
            String s = i2s.get(i);
            list.add(new Pair<String, Float>(s, p));
            // OOV
        }

        Collections.sort(list, new Comparator<Pair<String, Float>>() {
            public int compare(Pair<String, Float> o1, Pair<String, Float> o2) {
                return o2.second.compareTo(o1.second);
            }
        });

        for(Pair<String, Float> element : list) {
            if (result.size() >= topK) break;
            result.add(element);
        }

        return result;
    }
}
