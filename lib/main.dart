import 'package:flutter/material.dart';

import 'screens/home_screen.dart';
import 'services/threshold_config_service.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await ThresholdConfigService().loadConfig();
  runApp(const FitnessPoseApp());
}

class FitnessPoseApp extends StatelessWidget {
  const FitnessPoseApp({super.key});

  @override
  Widget build(BuildContext context) {
    const seed = Color(0xFF11998E);
    final scheme = ColorScheme.fromSeed(
      seedColor: seed,
      brightness: Brightness.light,
    );

    return MaterialApp(
      title: '智能健身教练',
      theme: ThemeData(
        colorScheme: scheme,
        useMaterial3: true,
        scaffoldBackgroundColor: const Color(0xFFF4F8F7),
        appBarTheme: const AppBarTheme(
          backgroundColor: Colors.transparent,
          elevation: 0,
          scrolledUnderElevation: 0,
          foregroundColor: Color(0xFF071A1B),
          centerTitle: true,
          titleTextStyle: TextStyle(
            color: Color(0xFF071A1B),
            fontSize: 20,
            fontWeight: FontWeight.w800,
          ),
        ),
        cardTheme: CardThemeData(
          color: Colors.white,
          elevation: 0,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(24),
          ),
        ),
        filledButtonTheme: FilledButtonThemeData(
          style: FilledButton.styleFrom(
            backgroundColor: const Color(0xFF0F766E),
            foregroundColor: Colors.white,
            padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 14),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(16),
            ),
            textStyle: const TextStyle(fontWeight: FontWeight.w700),
          ),
        ),
      ),
      debugShowCheckedModeBanner: false,
      home: const HomeScreen(),
    );
  }
}
