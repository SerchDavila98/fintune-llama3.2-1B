// lib/screens/add_chatbot_screen.dart

import 'dart:io';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:file_picker/file_picker.dart';
import 'package:uuid/uuid.dart';
import '../providers/chatbot_provider.dart';
import '../models/chatbot.dart';
import '../services/api_service.dart';

class AddChatbotScreen extends StatefulWidget {
  @override
  _AddChatbotScreenState createState() => _AddChatbotScreenState();
}

class _AddChatbotScreenState extends State<AddChatbotScreen> {
  final _formKey = GlobalKey<FormState>();
  String _name = '';
  String _useCase = '';
  List<File> _selectedFiles = [];
  bool _isLoading = false;

  final ApiService apiService = ApiService(baseUrl: 'http://your-backend-api.com');

  Future<void> _pickFiles() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['pdf'],
      allowMultiple: true,
    );

    if (result != null) {
      setState(() {
        _selectedFiles = result.paths.map((path) => File(path!)).toList();
      });
    }
  }

  Future<void> _submit() async {
    if (!_formKey.currentState!.validate()) return;

    if (_selectedFiles.isEmpty) {
      bool proceed = await showDialog(
        context: context,
        builder: (context) => AlertDialog(
          title: Text('No Files Selected'),
          content: Text('Are you sure you want to create a chatbot without uploading files?'),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context, false),
              child: Text('Cancel'),
            ),
            TextButton(
              onPressed: () => Navigator.pop(context, true),
              child: Text('Proceed'),
            ),
          ],
        ),
      );
      if (!proceed) return;
    }

    _formKey.currentState!.save();

    setState(() {
      _isLoading = true;
    });

    try {
      // Upload PDFs and generate dataset
      Map<String, dynamic> response = await apiService.uploadPdfs(
        useCase: _useCase,
        pdfFiles: _selectedFiles,
      );

      List<dynamic> dataset = response['dataset'];

      // Create Chatbot instance
      Chatbot newChatbot = Chatbot(
        id: Uuid().v4(),
        name: _name,
        useCase: _useCase,
        filePaths: _selectedFiles.map((file) => file.path).toList(),
        isLocal: true,
      );

      // Add to provider
      Provider.of<ChatbotProvider>(context, listen: false).addChatbot(newChatbot);

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Chatbot "${_name}" created successfully')),
      );

      Navigator.pop(context);
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error: ${e.toString()}')),
      );
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  Widget _buildFileList() {
    if (_selectedFiles.isEmpty) {
      return Text('No files selected.');
    }
    return Column(
      children: _selectedFiles
          .map((file) => ListTile(
                leading: Icon(Icons.picture_as_pdf, color: Colors.red),
                title: Text(file.path.split('/').last),
              ))
          .toList(),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        appBar: AppBar(
          title: Text('Add New Chatbot'),
        ),
        body: _isLoading
            ? Center(child: CircularProgressIndicator())
            : Padding(
                padding: EdgeInsets.all(16.0),
                child: Form(
                  key: _formKey,
                  child: SingleChildScrollView(
                    child: Column(
                      children: [
                        TextFormField(
                          decoration: InputDecoration(labelText: 'Chatbot Name'),
                          validator: (value) {
                            if (value == null || value.isEmpty) {
                              return 'Please enter a name';
                            }
                            return null;
                          },
                          onSaved: (value) => _name = value!,
                        ),
                        TextFormField(
                          decoration: InputDecoration(labelText: 'Use Case'),
                          validator: (value) {
                            if (value == null || value.isEmpty) {
                              return 'Please enter a use case';
                            }
                            return null;
                          },
                          onSaved: (value) => _useCase = value!,
                        ),
                        SizedBox(height: 20),
                        ElevatedButton.icon(
                          onPressed: _pickFiles,
                          icon: Icon(Icons.attach_file),
                          label: Text('Upload PDF Files'),
                        ),
                        SizedBox(height: 10),
                        _buildFileList(),
                        SizedBox(height: 30),
                        ElevatedButton(
                          onPressed: _submit,
                          child: Text('Create Chatbot'),
                          style: ElevatedButton.styleFrom(
                            minimumSize: Size(double.infinity, 50),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ));
  }
}
