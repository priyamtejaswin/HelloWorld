package com.example.helloworld;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.app.Activity;
import android.content.Intent;
import android.media.Image;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.provider.Settings;
import android.speech.tts.TextToSpeech;
import android.text.Editable;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.example.helloworld.ml.QaClient;
import com.github.dhaval2404.imagepicker.ImagePicker;

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
    ImageView imageView;
    Uri imageUri;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        answerView = (TextView) findViewById(R.id.answerView);
        inputText = (EditText) findViewById(R.id.editTextQuestion);
        imageView = (ImageView) findViewById(R.id.imageView2);
        imageView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                openImagePicker();
            }
        });

        qaClient = new QaClient(this);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == Activity.RESULT_OK) {
            //Image Uri will not be null for RESULT_OK
            Uri uri = data.getData();
            imageUri = uri;
            // Use Uri object instead of File to avoid storage permissions
            imageView.setImageURI(uri);
        } else if (resultCode == ImagePicker.RESULT_ERROR) {
            Toast.makeText(this, ImagePicker.getError(data), Toast.LENGTH_SHORT).show();
        } else {
            Toast.makeText(this, "Task Cancelled", Toast.LENGTH_SHORT).show();
        }
    }

    private void openImagePicker() {
        ImagePicker.with(this)
                .crop()	    			//Crop image(Optional), Check Customization for more option
                .compress(1024)			//Final image size will be less than 1 MB(Optional)
                .maxResultSize(1080, 1080)	//Final image resolution will be less than 1080 x 1080(Optional)
                .start();
    }

    public void updateText(View view) {
        Editable query = inputText.getText();
        String answer = qaClient.doVQA( imageUri, query.toString());



        int[][] bertids = qaClient.q2ids(query.toString());
        for (int element : bertids[0]) {
            System.out.println(element);
        }
        // answerView.setText("Processed! " + Arrays.toString(bertids[0]));
        answerView.setText("Answer: " + answer);
        Log.v(TAG, "QA complete.");
    }

    @Override
    protected void onStart() {
        Log.v(TAG, "onStart");
        super.onStart();
//        handler.post(() -> qaClient.loadDictionary());
        qaClient.loadDictionary();

        qaClient.loadModel();
        qaClient.loadImagePreprocessor();
        qaClient.loadVqaAns();
    }

    @Override
    protected void onStop() {
        Log.v(TAG, "onStop");
        super.onStop();
//        handler.post(() -> qaClient.unload());
        qaClient.unload();
    }
}