package com.example.helloworld;

import android.content.Context;
import androidx.test.platform.app.InstrumentationRegistry;
import androidx.test.ext.junit.runners.AndroidJUnit4;

import org.junit.Test;
import org.junit.runner.RunWith;

import static org.junit.Assert.*;

import com.example.helloworld.ml.QaClient;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Instrumented test, which will execute on an Android device.
 *
 * @see <a href="http://d.android.com/tools/testing">Testing documentation</a>
 */
@RunWith(AndroidJUnit4.class)
public class SanityTests {

    @Test
    public void textEncodingTest() {

        QaClient qaClient = new QaClient(
                InstrumentationRegistry.getInstrumentation().getTargetContext());

        qaClient.loadDictionary();

        qaClient.loadModel();
        qaClient.loadImagePreprocessor();
        qaClient.loadVqaAns();

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

        qaClient.unload();

    }
}