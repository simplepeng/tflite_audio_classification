package org.tensorflow.lite.examples.audio.fragments;

import androidx.annotation.NonNull;
import androidx.navigation.ActionOnlyNavDirections;
import androidx.navigation.NavDirections;
import org.tensorflow.lite.examples.audio.R;

public class PermissionsFragmentDirections {
  private PermissionsFragmentDirections() {
  }

  @NonNull
  public static NavDirections actionPermissionsToAudio() {
    return new ActionOnlyNavDirections(R.id.action_permissions_to_audio);
  }
}
