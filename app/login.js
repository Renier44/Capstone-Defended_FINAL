import React, { useState } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  Alert,
  Image,
} from 'react-native';
import { useRouter } from 'expo-router';
import * as SecureStore from 'expo-secure-store';
import { Ionicons } from '@expo/vector-icons';

const API_BASE_URL ='https://2b7bf55b1e09.ngrok-free.app/api';

export default function Login() {
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);

  const handleLogin = async () => {
    // NOTE: Alert is used here, but for production apps, use a custom UI modal instead of Alert.
    if (!email || !password) {
      Alert.alert('Missing Fields', 'Please enter both email and password.');
      return;
    }

    try {
      // =========================================================
      // ✅ FIX FOR 400 ERROR:
      // Map the user's email input to the 'username' key, 
      // as Django servers often require 'username' for the credential field.
      // =========================================================
      const payload = {
        username: email.trim(), // Assuming the server expects 'username' instead of 'email'
        password,
      };

      const response = await fetch(`${API_BASE_URL}/login/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      // =========================================================
      // ROBUST ERROR HANDLING
      // =========================================================

      if (!response.ok) {
        // Read the body as text to capture the error details (even if it's HTML/text)
        const errorText = await response.text();
        
        console.error('Server responded with status:', response.status);
        console.error('Raw error content (for backend debugging):', errorText.substring(0, 300) + '...');
        
        if (response.status === 400) {
            // Revert to a general message if the payload configuration failed
            throw new Error('Authentication failed. Please confirm your email/password.');
        } else if (response.status === 401 || response.status === 403) {
            // 401/403 typically means invalid credentials
            throw new Error('Invalid email or password. Please try again.');
        } else {
            // For 500 or other unexpected status codes
            throw new Error(`Server Error (${response.status}). Check console for details.`);
        }
      }

      // If we reach here, the status is 200-299, and it's safe to parse as JSON.
      const data = await response.json();
      console.log('Login response data:', data);

      if (data.token) {
        await SecureStore.setItemAsync('userToken', data.token);

        // =========================================================
        // Data Merging Logic to Preserve Local Profile Image
        // =========================================================

        const existingProfileStr = await SecureStore.getItemAsync('userProfile');
        const existingProfile = existingProfileStr ? JSON.parse(existingProfileStr) : {};

        const finalProfileToSave = {
            ...data, 
            profile_image: existingProfile.profile_image || data.profile_image, 
        };

        await SecureStore.setItemAsync('userProfile', JSON.stringify(finalProfileToSave));
        
        router.replace('/dashboard');
      } else {
        // Handle cases where response.ok is true, but 'data' indicates a logical failure
        Alert.alert('Login Failed', data.message || 'Invalid credentials or missing token.');
      }
    } catch (error) {
      // Catch network errors (connection failed) OR custom errors thrown above
      console.error('Login error:', error);
      // Show the error message to the user for better debugging
      Alert.alert('Login Error', error.message || 'Failed to connect to the server. Check your network.');
    }
  };

  return (
    <View style={styles.container}>
      {/* Logo + App Name beside each other */}
      <View style={styles.brandRow}>
        <Text style={styles.appName}>
          <Text style={styles.smart}>SMART </Text>
          <Text style={styles.sight}>SIGHT</Text>
        </Text>
        <Image
          source={require('../assets/images/icon.png')}
          style={styles.logo}
        />
      </View>

      <Text style={styles.subBrand}>Enhance Vision Optical PH</Text>

      {/* Instruction */}
      <Text style={styles.instruction}>sign in to access your account</Text>

      {/* Input Container */}
      <View style={styles.inputContainer}>
        <View style={styles.inputWrapper}>
          <TextInput
            style={styles.input}
            placeholder="Enter your email"
            placeholderTextColor="#666"
            autoCapitalize="none"
            keyboardType="email-address"
            value={email}
            onChangeText={setEmail}
          />
          <Ionicons name="mail-outline" size={20} color="#666" />
        </View>

        <View style={styles.inputWrapper}>
          <TextInput
            style={styles.input}
            placeholder="Password"
            placeholderTextColor="#666"
            secureTextEntry={!showPassword}
            value={password}
            onChangeText={setPassword}
          />
          <Ionicons name="lock-closed-outline" size={20} color="#666" />
        </View>

        <TouchableOpacity onPress={() => setShowPassword(!showPassword)}>
          <Text style={styles.showPassword}>
            {showPassword ? 'Hide password' : 'Show password'}
          </Text>
        </TouchableOpacity>
      </View>

      {/* Next Button */}
      <TouchableOpacity style={styles.button} onPress={handleLogin}>
        <Text style={styles.buttonText}>Next ›</Text>
      </TouchableOpacity>

      {/* Register Link */}
      <TouchableOpacity onPress={() => router.push('/register')}>
        <Text style={styles.register}>
          New Member? <Text style={styles.registerNow}>Register now</Text>
        </Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#77CDE0', // This was changed in the provided code block, reverting to original color from context
    padding: 30,
    justifyContent: 'center',
  },
  brandRow: {
    flexDirection: 'row', // text left, logo right
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 8,
  },
  appName: {
    fontSize: 40,
    fontWeight: 'bold',
    lineHeight: 40,
    textAlign: 'right',
    marginRight: 15,
  },
  smart: {
    color: '#FFFFFF', // white
  },
  sight: {
    color: '#FFD54F', // yellow
  },
  logo: {
    width: 70,
    height: 70,
    resizeMode: 'contain',
  },
  subBrand: {
    fontSize: 14,
    color: '#ddd',
    textAlign: 'center',
    marginBottom: 15,
  },
  instruction: {
    textAlign: 'center',
    color: '#555',
    marginBottom: 25,
    fontSize: 15,
  },
  inputContainer: {
    backgroundColor: 'rgba(255,255,255,0.45)',
    borderRadius: 20,
    padding: 15,
    marginBottom: 20,
  },
  inputWrapper: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#C8EAF7',
    borderRadius: 12,
    marginBottom: 15,
    paddingHorizontal: 15,
    height: 50,
  },
  input: { flex: 1, fontSize: 16 },
  showPassword: {
    textAlign: 'right',
    color: '#5B4FE9',
    fontSize: 13,
    marginTop: 5,
  },
  button: {
    backgroundColor: '#FFD54F',
    borderRadius: 25,
    height: 50,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 15,
  },
  buttonText: { color: '#fff', fontWeight: 'bold', fontSize: 18 },
  register: { textAlign: 'center', color: '#555', fontSize: 14 },
  registerNow: { color: '#5B4FE9', fontWeight: '600' },
});
