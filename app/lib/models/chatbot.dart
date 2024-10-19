// lib/models/chatbot.dart

import 'dart:convert';

class Chatbot {
  final String id;
  final String name;
  final String useCase;
  final List<String> filePaths;
  bool isLocal;

  Chatbot({
    required this.id,
    required this.name,
    required this.useCase,
    required this.filePaths,
    this.isLocal = true,
  });

  factory Chatbot.fromJson(Map<String, dynamic> json) {
    return Chatbot(
      id: json['id'],
      name: json['name'],
      useCase: json['useCase'],
      filePaths: List<String>.from(json['filePaths']),
      isLocal: json['isLocal'],
    );
  }

  Map<String, dynamic> toJson() => {
        'id': id,
        'name': name,
        'useCase': useCase,
        'filePaths': filePaths,
        'isLocal': isLocal,
      };

  static List<Chatbot> decode(String chatbots) =>
      (json.decode(chatbots) as List<dynamic>)
          .map<Chatbot>((item) => Chatbot.fromJson(item))
          .toList();

  static String encode(List<Chatbot> chatbots) => json.encode(
        chatbots
            .map<Map<String, dynamic>>((chatbot) => chatbot.toJson())
            .toList(),
      );
}
