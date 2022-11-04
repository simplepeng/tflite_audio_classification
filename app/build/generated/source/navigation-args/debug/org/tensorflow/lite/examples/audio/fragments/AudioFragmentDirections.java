package org.tensorflow.lite.examples.audio.fragments;

import androidx.annotation.NonNull;
import androidx.navigation.ActionOnlyNavDirections;
import androidx.navigation.NavDirections;
import org.tensorflow.lite.examples.audio.R;

public class AudioFragmentDirections {
  private AudioFragmentDirections() {
  }

  @NonNull
  public static NavDirections actionAudioToPermissions() {
    return new ActionOnlyNavDirections(R.id.action_audio_to_permissions);
  }
}
