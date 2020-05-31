/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.posenet

import android.os.Bundle
import android.support.v7.app.AppCompatActivity
import android.view.View
import android.widget.*
import kotlinx.android.synthetic.main.tfe_pn_activity_posenet.*

class CameraActivity : AppCompatActivity() {

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    setContentView(R.layout.tfe_pn_activity_camera)

//    val exercises: Array<String> = arrayOf("Squat", "Squat(side)", "Plank", "Side Plank",
//      "Push-Ups", "Hip Bridges", "Single-leg Hip Bridges")
//
//    val exercise_spinner = findViewById<Spinner>(R.id.exercise_spinner)
//    if(exercise_spinner != null){
//      val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, exercises)
//
//      exercise_spinner.adapter = adapter
//
//      exercise_spinner.onItemSelectedListener = object :
//        AdapterView.OnItemSelectedListener{
//        override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
//          Toast.makeText(this@CameraActivity, getString(R.string.selected_exercise) + exercises[position], Toast.LENGTH_LONG).show()
//        }
//
//        override fun onNothingSelected(parent: AdapterView<*>?) {
//          TODO("Not yet implemented")
//        }
//      }
//
//    }
    savedInstanceState ?: supportFragmentManager.beginTransaction()
      .replace(R.id.container, PosenetActivity())
      .commit()
  }
}
