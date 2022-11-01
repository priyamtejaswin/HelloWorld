package com.example.helloworld;

import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;
import android.os.Handler;
import android.provider.Settings;
import android.speech.tts.TextToSpeech;
import android.text.Editable;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import com.example.helloworld.ml.QaClient;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainApp";
//    Handler handler;
    private QaClient qaClient;

    TextView answerView;
    EditText inputText;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        answerView = (TextView) findViewById(R.id.answerView);
        inputText = (EditText) findViewById(R.id.editTextQuestion);
        qaClient = new QaClient(this);
    }

    public void updateText(View view) {
        Editable query = inputText.getText();

        qaClient.doVQA("helmet.jpg", query.toString());

        int[][] bertids = qaClient.q2ids(query.toString());
        for (int element : bertids[0]) {
            System.out.println(element);
        }
        System.out.println("Done!");
        answerView.setText("Processed! " + Arrays.toString(bertids[0]));
        System.out.println("Clicked.");
    }

    @Override
    protected void onStart() {
        Log.v(TAG, "onStart");
        super.onStart();
//        handler.post(() -> qaClient.loadDictionary());
        Log.v(TAG, "Calling loadDictionary");
        qaClient.loadDictionary();

        // Run all tests here for parity!
        String[] texts = {"is this pizza vegetarian?",
                "Is this pizza vegetarian?",
                "how many dogs are there in this picture ?",
                "How many Dogs are there in this picture ?",
                "what is next to the number 102"
        };
        int[][] truths = {{101, 2003, 2023, 10733, 23566, 1029, 102},
                {101, 2003, 2023, 10733, 23566, 1029, 102},
                {101, 2129, 2116, 6077, 2024, 2045, 1999, 2023, 3861, 1029, 102},
                {101, 2129, 2116, 6077, 2024, 2045, 1999, 2023, 3861, 1029, 102},
                {101, 2054, 2003, 2279, 2000, 1996, 2193, 9402, 102}
        };

        for (int i = 0; i < texts.length; i++) {
            String t = texts[i];
            List<Integer> pred = new ArrayList<>();
            int[] target = truths[i];

            for (Integer e : qaClient.q2ids(t)[0]) {
                if (e != 0) {
                    pred.add(e);
                    if (e == 102) break;
                } else break;
            }

            int[] ids = new int[0];
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.N) {
                ids = pred.stream().mapToInt(Integer::intValue).toArray();
            }

            System.out.println("p: " + Arrays.toString(ids));
            System.out.println("t: " + Arrays.toString(target));
            assert Arrays.equals(ids, target) : "LAST ONE FAILED!";
        }

        qaClient.loadModel();

        qaClient.loadImagePreprocessor();
    }

    @Override
    protected void onStop() {
        Log.v(TAG, "onStop");
        super.onStop();
//        handler.post(() -> qaClient.unload());
        qaClient.unload();
    }
}