import 'package:fitness_pose_app/models/threshold_profile.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('resolves bucket with group fallback order', () {
    final profile = ThresholdProfile.fromJson(<String, dynamic>{
      'schemaVersion': 2,
      'generatedAtIso': '2026-04-27T00:00:00.000Z',
      'exercises': <String, dynamic>{
        'squat': <String, dynamic>{
          'front': <String, dynamic>{
            'male': <String, dynamic>{
              'metrics': <String, dynamic>{
                'squatDownAngle': <String, dynamic>{'selected': 98},
              },
            },
            'default': <String, dynamic>{
              'metrics': <String, dynamic>{
                'squatDownAngle': <String, dynamic>{'selected': 100},
              },
            },
          },
        },
      },
    });

    final thresholds = profile.resolveThresholds(
      exercise: 'squat',
      view: 'front',
      groups: const <String>['male_medium', 'male', 'medium', 'default'],
    );

    expect(thresholds['squatDownAngle'], 98);
  });

  test('parses legacy flat schema', () {
    final profile = ThresholdProfile.fromJson(<String, dynamic>{
      'squat': <String, dynamic>{
        'front': <String, dynamic>{
          'default': <String, dynamic>{
            'maxTorsoLeanDeg': 20,
          },
        },
      },
    });

    final thresholds = profile.resolveThresholds(
      exercise: 'squat',
      view: 'front',
      groups: const <String>['default'],
    );

    expect(thresholds['maxTorsoLeanDeg'], 20);
  });

  test('default profile includes expanded data-driven metrics', () {
    final thresholds = ThresholdProfile.defaultProfile().resolveThresholds(
      exercise: 'pushup',
      view: 'side',
      groups: const <String>['default'],
    );

    expect(thresholds, containsPair('pushupDepthMargin', 10));
    expect(thresholds, containsPair('pushupHipSagOffset', -0.045));
    expect(thresholds, containsPair('maxElbowFlareDeg', 72));
  });
}
