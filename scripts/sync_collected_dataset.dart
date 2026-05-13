import 'dart:convert';
import 'dart:io';

Future<void> main(List<String> args) async {
  final options = _CliOptions.parse(args);
  final sourceDirs = options.sourceDirs.isEmpty
      ? _defaultSourceDirectories()
      : options.sourceDirs.map((path) => Directory(path)).toList();

  final existingSources = <Directory>[];
  for (final dir in sourceDirs) {
    if (await dir.exists()) {
      existingSources.add(dir);
    }
  }

  if (existingSources.isEmpty) {
    stderr.writeln(
      'No dataset directories found. '
      'Pass `--source-dir <path>` or copy collected JSONL files into '
      '`data/dataset/import`.',
    );
    exitCode = 2;
    return;
  }

  final repoDatasetDir = options.outputFile.parent;
  await repoDatasetDir.create(recursive: true);

  final inputFiles = await _collectInputFiles(
    sourceDirs: existingSources,
    outputFile: options.outputFile,
  );

  if (inputFiles.isEmpty) {
    stderr.writeln(
      'No `fitness_dataset*.jsonl` files were found in:\n'
      '${existingSources.map((dir) => ' - ${dir.path}').join('\n')}',
    );
    exitCode = 2;
    return;
  }

  final merger = _DatasetMerger();
  final report = await merger.merge(
    inputFiles: inputFiles,
    outputFile: options.outputFile,
    reportFile: options.reportFile,
  );

  stdout.writeln('Merged dataset written to ${options.outputFile.path}');
  stdout.writeln('Sync report written to ${options.reportFile.path}');
  stdout.writeln(
    'Input files: ${report.inputFileCount}, '
    'raw lines: ${report.totalInputLines}, '
    'output samples: ${report.outputSampleCount}',
  );
}

class _CliOptions {
  const _CliOptions({
    required this.sourceDirs,
    required this.outputFile,
    required this.reportFile,
  });

  final List<String> sourceDirs;
  final File outputFile;
  final File reportFile;

  static _CliOptions parse(List<String> args) {
    final sourceDirs = <String>[];
    var outputPath = 'data/dataset/fitness_dataset.jsonl';
    var reportPath = 'data/dataset/fitness_dataset_sync.report.json';

    for (var i = 0; i < args.length; i++) {
      final arg = args[i];
      switch (arg) {
        case '--source-dir':
          if (i + 1 >= args.length) {
            _fail('Missing value for --source-dir');
          }
          sourceDirs.add(args[++i]);
          break;
        case '--output':
          if (i + 1 >= args.length) {
            _fail('Missing value for --output');
          }
          outputPath = args[++i];
          break;
        case '--report':
          if (i + 1 >= args.length) {
            _fail('Missing value for --report');
          }
          reportPath = args[++i];
          break;
        case '--help':
        case '-h':
          _printHelp();
          exit(0);
        default:
          _fail('Unknown argument: $arg');
      }
    }

    return _CliOptions(
      sourceDirs: sourceDirs,
      outputFile: File(outputPath),
      reportFile: File(reportPath),
    );
  }

  static Never _fail(String message) {
    stderr.writeln(message);
    _printHelp();
    exit(64);
  }

  static void _printHelp() {
    stdout.writeln('Usage: dart run scripts/sync_collected_dataset.dart [options]');
    stdout.writeln('');
    stdout.writeln('Options:');
    stdout.writeln(
      '  --source-dir <path>   Add a dataset directory to import. '
      'Can be repeated.',
    );
    stdout.writeln(
      '  --output <path>       Merged JSONL output. '
      '(default: data/dataset/fitness_dataset.jsonl)',
    );
    stdout.writeln(
      '  --report <path>       Sync report JSON. '
      '(default: data/dataset/fitness_dataset_sync.report.json)',
    );
  }
}

List<Directory> _defaultSourceDirectories() {
  final env = Platform.environment;
  final dirs = <Directory>[
    Directory('data/dataset'),
    Directory('data/dataset/import'),
  ];

  void addIfNotEmpty(String? basePath, List<String> parts) {
    if (basePath == null || basePath.trim().isEmpty) {
      return;
    }
    final path = [basePath, ...parts].join(Platform.pathSeparator);
    dirs.add(Directory(path));
  }

  addIfNotEmpty(env['APPDATA'], <String>['fitness_pose_app', 'dataset']);
  addIfNotEmpty(env['LOCALAPPDATA'], <String>['fitness_pose_app', 'dataset']);
  addIfNotEmpty(env['USERPROFILE'], <String>['Documents', 'dataset']);
  addIfNotEmpty(
    env['HOME'],
    <String>['Documents', 'dataset'],
  );
  addIfNotEmpty(
    env['HOME'],
    <String>['Library', 'Application Support', 'fitness_pose_app', 'dataset'],
  );
  addIfNotEmpty(
    env['HOME'],
    <String>['.local', 'share', 'fitness_pose_app', 'dataset'],
  );

  final unique = <String>{};
  return dirs.where((dir) => unique.add(dir.path)).toList();
}

Future<List<File>> _collectInputFiles({
  required List<Directory> sourceDirs,
  required File outputFile,
}) async {
  final files = <String, File>{};

  Future<void> collectFromDirectory(Directory dir) async {
    await for (final entity in dir.list(recursive: false, followLinks: false)) {
      if (entity is! File) {
        continue;
      }
      final name = entity.uri.pathSegments.isEmpty
          ? entity.path
          : entity.uri.pathSegments.last;
      final lowerName = name.toLowerCase();
      if (!lowerName.endsWith('.jsonl')) {
        continue;
      }
      if (!lowerName.startsWith('fitness_dataset')) {
        continue;
      }
      files[entity.absolute.path] = entity;
    }
  }

  for (final dir in sourceDirs) {
    await collectFromDirectory(dir);
  }

  if (await outputFile.exists()) {
    files[outputFile.absolute.path] = outputFile;
  }

  final collected = files.values.toList()
    ..sort((a, b) => a.path.compareTo(b.path));
  return collected;
}

class _DatasetSyncReport {
  const _DatasetSyncReport({
    required this.inputFileCount,
    required this.totalInputLines,
    required this.invalidJsonLines,
    required this.mergedDuplicates,
    required this.outputSampleCount,
    required this.inputFiles,
    required this.exerciseCounts,
    required this.viewCounts,
    required this.qualityCounts,
    required this.subjectCount,
  });

  final int inputFileCount;
  final int totalInputLines;
  final int invalidJsonLines;
  final int mergedDuplicates;
  final int outputSampleCount;
  final List<String> inputFiles;
  final Map<String, int> exerciseCounts;
  final Map<String, int> viewCounts;
  final Map<String, int> qualityCounts;
  final int subjectCount;

  Map<String, dynamic> toJson() {
    return <String, dynamic>{
      'generatedAtIso': DateTime.now().toIso8601String(),
      'inputFileCount': inputFileCount,
      'totalInputLines': totalInputLines,
      'invalidJsonLines': invalidJsonLines,
      'mergedDuplicates': mergedDuplicates,
      'outputSampleCount': outputSampleCount,
      'inputFiles': inputFiles,
      'exerciseCounts': exerciseCounts,
      'viewCounts': viewCounts,
      'qualityCounts': qualityCounts,
      'subjectCount': subjectCount,
    };
  }
}

class _DatasetMerger {
  Future<_DatasetSyncReport> merge({
    required List<File> inputFiles,
    required File outputFile,
    required File reportFile,
  }) async {
    final mergedById = <String, Map<String, dynamic>>{};
    var totalInputLines = 0;
    var invalidJsonLines = 0;
    var mergedDuplicates = 0;

    for (final file in inputFiles) {
      final lines = await file.readAsLines();
      for (final rawLine in lines) {
        final line = rawLine.trim();
        if (line.isEmpty) {
          continue;
        }
        totalInputLines += 1;
        final decoded = _decodeMap(line);
        if (decoded == null) {
          invalidJsonLines += 1;
          continue;
        }

        final id = _recordIdentity(decoded);
        final current = mergedById[id];
        if (current == null) {
          mergedById[id] = decoded;
          continue;
        }

        mergedDuplicates += 1;
        mergedById[id] = _preferBetterRecord(current, decoded);
      }
    }

    final records = mergedById.values.toList()
      ..sort(_compareRecordsByTime);

    await outputFile.parent.create(recursive: true);
    final sink = outputFile.openWrite();
    for (final record in records) {
      sink.writeln(jsonEncode(record));
    }
    await sink.flush();
    await sink.close();

    final exerciseCounts = <String, int>{};
    final viewCounts = <String, int>{};
    final qualityCounts = <String, int>{};
    final subjects = <String>{};

    for (final record in records) {
      _increment(exerciseCounts, record['exercise_type']?.toString() ?? 'unknown');
      _increment(viewCounts, record['view_tag']?.toString() ?? 'unknown');
      _increment(qualityCounts, record['quality_tag']?.toString() ?? 'unknown');
      final subject = record['subject_tag']?.toString();
      if (subject != null && subject.trim().isNotEmpty) {
        subjects.add(subject);
      }
    }

    final report = _DatasetSyncReport(
      inputFileCount: inputFiles.length,
      totalInputLines: totalInputLines,
      invalidJsonLines: invalidJsonLines,
      mergedDuplicates: mergedDuplicates,
      outputSampleCount: records.length,
      inputFiles: inputFiles.map((file) => file.path).toList(),
      exerciseCounts: exerciseCounts,
      viewCounts: viewCounts,
      qualityCounts: qualityCounts,
      subjectCount: subjects.length,
    );

    await reportFile.parent.create(recursive: true);
    const encoder = JsonEncoder.withIndent('  ');
    await reportFile.writeAsString('${encoder.convert(report.toJson())}\n');

    return report;
  }

  Map<String, dynamic>? _decodeMap(String line) {
    try {
      final decoded = jsonDecode(line);
      if (decoded is Map<String, dynamic>) {
        return decoded;
      }
      if (decoded is Map) {
        return Map<String, dynamic>.from(decoded);
      }
    } catch (_) {
      return null;
    }
    return null;
  }

  String _recordIdentity(Map<String, dynamic> record) {
    final sampleId = record['sample_id']?.toString().trim();
    if (sampleId != null && sampleId.isNotEmpty) {
      return sampleId;
    }

    final parts = <String>[
      record['capture_time_left']?.toString() ?? record['timestamp']?.toString() ?? '',
      record['exercise_type']?.toString() ?? '',
      record['subject_tag']?.toString() ?? '',
      record['frame_index']?.toString() ?? '',
      record['video_path']?.toString() ?? '',
    ];
    return parts.join('|');
  }

  Map<String, dynamic> _preferBetterRecord(
    Map<String, dynamic> current,
    Map<String, dynamic> candidate,
  ) {
    final currentScore = _recordScore(current);
    final candidateScore = _recordScore(candidate);
    if (candidateScore > currentScore) {
      return candidate;
    }
    if (candidateScore < currentScore) {
      return current;
    }

    final currentTime = _recordTime(current);
    final candidateTime = _recordTime(candidate);
    if (candidateTime != null && currentTime != null) {
      return candidateTime.isAfter(currentTime) ? candidate : current;
    }
    return current;
  }

  int _recordScore(Map<String, dynamic> record) {
    var score = 0;
    if (record['landmarks_complete'] == true) {
      score += 100;
    }
    if (record['manual_checked'] == true) {
      score += 35;
    }

    final landmarksFinal = record['landmarks_final'];
    if (landmarksFinal is Map) {
      score += landmarksFinal.length;
    }

    return score;
  }

  DateTime? _recordTime(Map<String, dynamic> record) {
    final raw = record['capture_time_left']?.toString() ??
        record['timestamp']?.toString();
    if (raw == null || raw.isEmpty) {
      return null;
    }
    return DateTime.tryParse(raw);
  }

  int _compareRecordsByTime(
    Map<String, dynamic> left,
    Map<String, dynamic> right,
  ) {
    final leftTime = _recordTime(left);
    final rightTime = _recordTime(right);
    if (leftTime != null && rightTime != null) {
      final diff = leftTime.compareTo(rightTime);
      if (diff != 0) {
        return diff;
      }
    }
    final leftId = left['sample_id']?.toString() ?? '';
    final rightId = right['sample_id']?.toString() ?? '';
    return leftId.compareTo(rightId);
  }
}

void _increment(Map<String, int> map, String key) {
  final normalized = key.trim().isEmpty ? 'unknown' : key.trim();
  map[normalized] = (map[normalized] ?? 0) + 1;
}
