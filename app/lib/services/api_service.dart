// lib/services/api_service.dart

import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

class ApiService {
  final String baseUrl;

  ApiService({required this.baseUrl});

  // Endpoint to upload PDFs and generate dataset
  Future<Map<String, dynamic>> uploadPdfs({
    required String useCase,
    required List<File> pdfFiles,
  }) async {
    var uri = Uri.parse('$baseUrl/finetuning-rag/upload-pdfs');
    var request = http.MultipartRequest('POST', uri);

    request.fields['use_case'] = useCase;

    for (var file in pdfFiles) {
      request.files.add(await http.MultipartFile.fromPath(
        'files',
        file.path,
        filename: file.path.split('/').last,
      ));
    }

    var streamedResponse = await request.send();
    var response = await http.Response.fromStream(streamedResponse);

    if (response.statusCode == 200) {
      return json.decode(response.body);
    } else {
      throw Exception('Failed to upload PDFs: ${response.body}');
    }
  }

  // Endpoint to list chatbots (if managed via backend)
  // Implement additional API methods as needed
}
