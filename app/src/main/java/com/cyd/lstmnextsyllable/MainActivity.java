package com.cyd.lstmnextsyllable;

import static java.util.Arrays.*;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.util.Pair;

import android.content.res.AssetFileDescriptor;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.util.List;

public class MainActivity extends AppCompatActivity {
    private static final String LOG_TAG = "MainActivity";
    PredictModel syllabeModel = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        if (syllabeModel == null) {
            try {
                syllabeModel = new PredictModel(this, 10, "lstmmodel.tflite", "dict.csv");
            } catch(Exception e) {
                Log.d(LOG_TAG, e.getMessage());
            }
        }

        final EditText edit = (EditText) findViewById(R.id.editSyl);
        final TextView outView = (TextView) findViewById(R.id.outText);
        Button btnChange = (Button) findViewById(R.id.button);

        btnChange.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                if (syllabeModel != null) {
                    String inputText = edit.getText().toString();

                    List<Pair<String, Float>> predictions = syllabeModel.predict(edit.getText().toString());
                    for(Pair<String, Float> e : predictions) {
                        Log.d(LOG_TAG, e.first + "=>" + e.second);
                    }
                }
            }

        });
    }

    private Interpreter getTfliteInterpreter(String modelPath) {
        try {
            return new Interpreter(loadModelFile(MainActivity.this, modelPath));
        }
        catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    private MappedByteBuffer loadModelFile(MainActivity activity, String modelPath) throws IOException, IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private float[][][] makeOneHotEncding(int[] input) {
        float[][][] encodedData = new float[1][3][972];
        for (int i=0; i<1; i++)
            for (int j=0; j<3; j++)
                for (int k=0; k<972; k++)
                    encodedData[i][j][k] = 0.0f;

        encodedData[0][0][input[0]] = 1.0f;
        encodedData[0][1][input[1]] = 1.0f;
        encodedData[0][2][input[2]] = 1.0f;


//        int cnt = 0;
//        for (int i=0; i<972; i++) {
//
//            if (encodedData[0][2][i] == 0.0)
//                cnt++;
//            else
//                Log.d("TLite", encodedData[0][2][i] + " " + i + " th");
//        }
//        Log.d("TLite", cnt + " ");


        return encodedData;
    }




}