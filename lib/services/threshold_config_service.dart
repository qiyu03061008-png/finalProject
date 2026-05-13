import 'dart:convert';

import 'package:flutter/services.dart';

import '../models/threshold_profile.dart';

class ThresholdConfigService {
  static final ThresholdConfigService _instance =
      ThresholdConfigService._private();

  factory ThresholdConfigService() => _instance;

  ThresholdConfigService._private();

  ThresholdProfile _thresholdProfile = ThresholdProfile.defaultProfile();
  Future<void>? _loadingFuture;
  bool _isLoaded = false;

  Future<void> loadConfig() {
    final pending = _loadingFuture;
    if (pending != null) {
      return pending;
    }

    final future = _loadInternal();
    _loadingFuture = future;
    return future.whenComplete(() {
      _loadingFuture = null;
    });
  }

  Future<void> _loadInternal() async {
    try {
      final content =
          await rootBundle.loadString('assets/threshold_profile.json');
      final json = jsonDecode(content) as Map<String, dynamic>;
      _thresholdProfile = ThresholdProfile.fromJson(json);
    } catch (error) {
      _thresholdProfile = ThresholdProfile.defaultProfile();
      print('Failed to load threshold config: $error');
    } finally {
      _isLoaded = true;
    }
  }

  ThresholdProfile getSyncThresholdProfile() {
    if (!_isLoaded) {
      _thresholdProfile = ThresholdProfile.defaultProfile();
      _isLoaded = true;
      loadConfig();
    }
    return _thresholdProfile;
  }

  ThresholdProfile get thresholdProfile => getSyncThresholdProfile();

  Future<ThresholdProfile> ensureLoaded() async {
    await loadConfig();
    return _thresholdProfile;
  }

  List<String> getGroupFallbacks(double heightCm, String? gender) {
    final normalizedGender = _normalizeGender(gender);
    final heightGroup = _heightGroup(heightCm);
    final groups = <String>[];

    if (normalizedGender != null && heightGroup != null) {
      groups.add('${normalizedGender}_$heightGroup');
    }
    if (normalizedGender != null) {
      groups.add(normalizedGender);
    }
    if (heightGroup != null) {
      groups.add(heightGroup);
    }
    groups.add('default');

    return groups;
  }

  Map<String, double> getThresholds(
    String exercise,
    String view,
    double heightCm,
    String? gender,
  ) {
    return thresholdProfile.resolveThresholds(
      exercise: exercise,
      view: view,
      groups: getGroupFallbacks(heightCm, gender),
    );
  }

  ThresholdBucketProfile? getThresholdBucket(
    String exercise,
    String view,
    double heightCm,
    String? gender,
  ) {
    return thresholdProfile.resolveBucket(
      exercise: exercise,
      view: view,
      groups: getGroupFallbacks(heightCm, gender),
    );
  }

  String? _normalizeGender(String? gender) {
    final normalized = gender?.trim().toLowerCase();
    if (normalized == null || normalized.isEmpty) {
      return null;
    }
    if (normalized == 'male' || normalized == 'female') {
      return normalized;
    }
    return null;
  }

  String? _heightGroup(double heightCm) {
    if (heightCm < 160) {
      return 'short';
    }
    if (heightCm > 180) {
      return 'tall';
    }
    return 'medium';
  }
}
