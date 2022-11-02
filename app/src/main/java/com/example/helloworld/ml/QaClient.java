/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
package com.example.helloworld.ml;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.JsonReader;
import android.util.Log;

import androidx.annotation.WorkerThread;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.base.Joiner;

//import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.pytorch.IValue;
import org.pytorch.MemoryFormat;
import org.pytorch.Module;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Tensor;
 import org.pytorch.torchvision.TensorImageUtils;

/** Interface to load TfLite model and provide predictions. */
public class QaClient {
  private static final String TAG = "QaClient";
  private static final String DIC_PATH = "vocab.txt";
  private static final String MODEL_PATH = "optTracedVilt.ptl";
  private static final String PBT_PATH = "optPixelbertTransform.ptl";
  private static final String VQA_PATH = "vqa_dict.json";

  private static final int MAX_ANS_LEN = 32;
  private static final int MAX_QUERY_LEN = 64;
  private static final int MAX_SEQ_LEN = 384;
  private static final boolean DO_LOWER_CASE = true;
  private static final int PREDICT_ANS_NUM = 5;
  private static final int NUM_LITE_THREADS = 4;

  // Need to shift 1 for outputs ([CLS]).
  private static final int OUTPUT_OFFSET = 1;

  private final Context context;
  private final Map<String, Integer> dic = new HashMap<>();
  private final FeatureConverter featureConverter;
//  private Interpreter tflite;

  Module model;
  Module pbt;
  float[] ZERO_MEAN = new float[] {0.0f, 0.0f, 0.0f};
  float[] UNIT_STD = new float[] {1.0f, 1.0f, 1.0f};

  ObjectMapper mapper = new ObjectMapper();
  Map<Integer, String> vqa_ans = new HashMap<>();

  private static final Joiner SPACE_JOINER = Joiner.on(" ");

  public QaClient(Context context) {
    this.context = context;
    this.featureConverter = new FeatureConverter(dic, DO_LOWER_CASE, MAX_QUERY_LEN, MAX_SEQ_LEN);
  }

//  @WorkerThread
  public void loadDictionary() {
    try {
      Log.v(TAG, "Loading dictionary.");
      loadDictionaryFile(this.context.getAssets());
      Log.v(TAG, "Dictionary loaded.");
    } catch (IOException ex) {
      Log.e(TAG, ex.getMessage());
    }
  }

  public void loadVqaAns() {
    try {
      Log.v(TAG, "Loading VQA ans json.");
      loadVqaDictFile(this.context.getAssets());
      Log.v(TAG, "VQA ans json loaded.");
    } catch (IOException ex) {
      Log.e(TAG, ex.getMessage());
    }
  }

  public void loadModel() {
    try {
      Log.v(TAG, "Loading model.");
      loadModelFile(this.context);
      Log.v(TAG, "Model loaded.");
    } catch (IOException ex) {
      Log.e(TAG, ex.getMessage());
    }
  }

  public void loadImagePreprocessor() {
    try {
      Log.v(TAG, "Loading PixelBert preprocessor.");
      loadPBTFile(this.context);
      Log.v(TAG, "PixelBert preprocessor loaded.");
    } catch (IOException ex) {
      Log.e(TAG, ex.getMessage());
    }
  }

//  @WorkerThread
  public void unload() {
    dic.clear();
    model.destroy();
    pbt.destroy();
    vqa_ans.clear();
  }

  /** Load dictionary from assets. */
  public void loadDictionaryFile(AssetManager assetManager) throws IOException {
    Log.v(TAG, "Trying to load dictionary file.");
    try (InputStream ins = assetManager.open(DIC_PATH);
        BufferedReader reader = new BufferedReader(new InputStreamReader(ins))) {
      int index = 0;
      while (reader.ready()) {
        String key = reader.readLine();
        dic.put(key, index++);
      }
    }
  }

  public void loadModelFile(Context context) throws IOException {
    model = LiteModuleLoader.load(assetFilePath(context, MODEL_PATH));
  }

  public void loadPBTFile(Context context) throws IOException {
    pbt = LiteModuleLoader.load(assetFilePath(context, PBT_PATH));
  }

  public void loadVqaDictFile(AssetManager assetManager) throws IOException {
    Log.v(TAG, "Loading VQA answer dictionary.");
    try (InputStream ins = assetManager.open(VQA_PATH);) {
      JsonReader reader = new JsonReader(new InputStreamReader(ins));
      reader.beginObject();
      while (reader.hasNext()) {
        Integer key = Integer.parseInt(reader.nextName());
        String value = reader.nextString();
        vqa_ans.put(key, value);
      }
      reader.endObject();
      Log.v(TAG, "Items found:" + vqa_ans.size());
    }
  }

  /**
   * Input: Original content and query for the QA task. Later converted to Feature by
   * FeatureConverter. Output: A String[] array of answers and a float[] array of corresponding
   */
  @WorkerThread
  public synchronized int[][] convert(String query, String content) {
    Log.v(TAG, "Convert Feature...");
    Feature feature = featureConverter.convert(query, content);

    Log.v(TAG, "Set inputs...");
    int[][] inputIds = new int[1][MAX_SEQ_LEN];
//    int[][] inputMask = new int[1][MAX_SEQ_LEN];
//    int[][] segmentIds = new int[1][MAX_SEQ_LEN];

    for (int j = 0; j < MAX_SEQ_LEN; j++) {
      inputIds[0][j] = feature.inputIds[j];
//      inputMask[0][j] = feature.inputMask[j];
//      segmentIds[0][j] = feature.segmentIds[j];
    }
    return inputIds;
  }

  public int[][] q2ids(String query) {
    Log.v(TAG, "Converting query: " + query);
    Feature feature = featureConverter.convert(query, "");

    Log.v(TAG, "Set inputs...");
    int[][] inputIds = new int[1][MAX_SEQ_LEN];

    for (int j = 0; j < MAX_SEQ_LEN; j++) {
      inputIds[0][j] = feature.inputIds[j];
    }

    return inputIds;
  }

  /** Convert the answer back to original text form. */
  @WorkerThread
  private static String convertBack(Feature feature, int start, int end) {
     // Shifted index is: index of logits + offset.
    int shiftedStart = start + OUTPUT_OFFSET;
    int shiftedEnd = end + OUTPUT_OFFSET;
    int startIndex = feature.tokenToOrigMap.get(shiftedStart);
    int endIndex = feature.tokenToOrigMap.get(shiftedEnd);
    // end + 1 for the closed interval.
    String ans = SPACE_JOINER.join(feature.origTokens.subList(startIndex, endIndex + 1));
    return ans;
  }

  /** Get absolute path. */
  public static String assetFilePath(Context context, String assetName) throws IOException {
    File file = new File(context.getFilesDir(), assetName);
    if (file.exists() && file.length() > 0) {
      return file.getAbsolutePath();
    }

    try (InputStream is = context.getAssets().open(assetName)) {
      try (OutputStream os = new FileOutputStream(file)) {
        byte[] buffer = new byte[4 * 1024];
        int read;
        while ((read = is.read(buffer)) != -1) {
          os.write(buffer, 0, read);
        }
        os.flush();
      }
      return file.getAbsolutePath();
    }
  }

  public String doVQA(String imgName, String question) {
    try {
      // Load image.
      Bitmap bitmap = BitmapFactory.decodeStream(this.context.getAssets().open(imgName));
      Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
              ZERO_MEAN,
              UNIT_STD,
              MemoryFormat.CHANNELS_LAST);
      Log.v(TAG, "Loaded image: " + Arrays.toString(inputTensor.shape()));

      // Pre-process question.
      List<Integer> input_ids = new ArrayList<>();
      for (Integer e : q2ids(question)[0]) {
        if (e != 0) {
          input_ids.add(e);
          if (e == 102) break;
        } else break;
      }

      int[] ids = new int[0];
      if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.N) {
        ids = input_ids.stream().mapToInt(Integer::intValue).toArray();
      }

      long[] shape = new long[] {1, ids.length};
      Tensor ts_ii = Tensor.fromBlob(ids, shape);

      int[] token_type = new int[ids.length];
      Arrays.fill(token_type, 0);
      Tensor ts_tt = Tensor.fromBlob(token_type, shape);

      int[] atten_mask = new int[ids.length];
      Arrays.fill(atten_mask, 1);
      Tensor ts_am = Tensor.fromBlob(atten_mask, shape);

      // Pre-process image.
      final Tensor img = pbt.forward(IValue.from(inputTensor)).toTensor();

      // Create batch.
      Map<String, IValue> batch = new HashMap<>();
      batch.put("text", IValue.from(ts_ii));
      batch.put("text_ids", IValue.from(ts_ii));
      batch.put("text_labels", IValue.from(ts_ii));
      batch.put("text_masks", IValue.from(ts_am));
      batch.put("image", IValue.from(img));

      final Tensor outputTensor = model.forward(IValue.dictStringKeyFrom(batch)).toTensor();
      Log.v(TAG, "Completed VQA task!");

      final float[] scores = outputTensor.getDataAsFloatArray();
      // Perform argmax manually.
      float maxScore = -Float.MAX_VALUE;
      int maxScoreIdx = -1;
      for (int i = 0; i < scores.length; i++) {
        if (scores[i] > maxScore) {
          maxScore = scores[i];
          maxScoreIdx = i;
        }
      }
      Log.v(TAG, "maxScore:" + maxScore);
      Log.v(TAG, "Index:" + maxScoreIdx);
      String answer = vqa_ans.get(maxScoreIdx);
      Log.v(TAG, "Answer:" + answer);
      return answer;
    }
    catch (IOException ex) {
      Log.e(TAG, ex.getMessage());
    }
    return "Failed.";
  }
}
