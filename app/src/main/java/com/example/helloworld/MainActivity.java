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
        int[][] bertids = qaClient.q2ids(query.toString());
        answerView.setText(bertids[0].toString());
        System.out.println(bertids[0]);
        System.out.println("Clicked.");
    }

    @Override
    protected void onStart() {
        Log.v(TAG, "onStart");
        super.onStart();
//        handler.post(() -> qaClient.loadDictionary());
        Log.v(TAG, "Calling loadDictionary");
        qaClient.loadDictionary();
    }

    @Override
    protected void onStop() {
        Log.v(TAG, "onStop");
        super.onStop();
//        handler.post(() -> qaClient.unload());
        qaClient.unload();
    }
}