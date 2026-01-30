import { MaterialIcons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { useState } from 'react';
import {
  Alert,
  KeyboardAvoidingView,
  Platform,
  SafeAreaView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
  Image,
} from 'react-native';
import * as SecureStore from 'expo-secure-store';

const API_BASE_URL ='https://2b7bf55b1e09.ngrok-free.app/api';

export default function RegisterScreen() {
  const router = useRouter();
  const [firstName, setFirstName] = useState('');
  const [lastName, setLastName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);

  const handleRegister = async () => {
    if (!firstName || !lastName || !email || !password) {
      Alert.alert('Error', 'Please fill out all fields.');
      return;
    }

    try {
    const response = await fetch(`${API_BASE_URL}/api/register/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      first_name: firstName.trim(),
      last_name: lastName.trim(),
      email: email.trim(),
      password: password.trim(),
    }),
});

      const data = await response.json();
      console.log('Register response:', response.status, data);

      if (!response.ok) {
        const errorMsg =
          data.errors
            ? Object.values(data.errors).flat().join('\n')
            : data.message || 'Something went wrong.';
        Alert.alert('Registration Failed', errorMsg);
        return;
      }

      if (data.token) {
        await SecureStore.setItemAsync('userToken', data.token);
      }

      if (data.user) {
        await SecureStore.setItemAsync('userProfile', JSON.stringify(data.user));
      }

      Alert.alert('Success', 'Registration successful!');
      router.replace('/login');
    } catch (error) {
      console.error('Register error:', error);
      Alert.alert('Error', 'Unable to register. Please check your network.');
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <KeyboardAvoidingView
        style={{ flex: 1 }}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      >
        {/* Back Button */}
        <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
          <MaterialIcons name="arrow-back-ios" size={24} color="#333" />
          <Text style={styles.backButtonText}>Back</Text>
        </TouchableOpacity>

        <View style={styles.content}>
          {/* Branding */}
          <View style={styles.brandRow}>
            <Text style={styles.smartText}>SMART</Text>
            <Text style={styles.sightText}>SIGHT</Text>
            {/* ✅ Logo only, no white circle */}
            <Image source={require('../assets/images/icon.png')} style={styles.logo} />
          </View>
          <Text style={styles.subBrand}>Enhance Vision Optical PH</Text>

          {/* Input Group */}
          <View style={styles.inputGroup}>
            {/* First Name */}
            <View style={styles.inputContainer}>
              <TextInput
                style={styles.input}
                placeholder="First Name"
                autoCapitalize="words"
                value={firstName}
                onChangeText={setFirstName}
              />
              <MaterialIcons name="person-outline" size={24} color="#555" style={styles.icon} />
            </View>

            {/* Last Name */}
            <View style={styles.inputContainer}>
              <TextInput
                style={styles.input}
                placeholder="Last Name"
                autoCapitalize="words"
                value={lastName}
                onChangeText={setLastName}
              />
              <MaterialIcons name="person-outline" size={24} color="#555" style={styles.icon} />
            </View>

            {/* Email */}
            <View style={styles.inputContainer}>
              <TextInput
                style={styles.input}
                placeholder="Email"
                keyboardType="email-address"
                autoCapitalize="none"
                value={email}
                onChangeText={setEmail}
              />
              <MaterialIcons name="mail-outline" size={24} color="#555" style={styles.icon} />
            </View>

            {/* Password */}
            <View style={styles.inputContainer}>
              <TextInput
                style={styles.input}
                placeholder="Password"
                secureTextEntry={!showPassword}
                value={password}
                onChangeText={setPassword}
              />
              <TouchableOpacity onPress={() => setShowPassword(!showPassword)} style={styles.icon}>
                <MaterialIcons
                  name={showPassword ? 'lock-open' : 'lock'}
                  size={24}
                  color="#555"
                />
              </TouchableOpacity>
            </View>
          </View>

          {/* Show/Hide Password */}
          <TouchableOpacity onPress={() => setShowPassword(!showPassword)}>
            <Text style={styles.showPasswordText}>
              {showPassword ? 'Hide password' : 'Show password'}
            </Text>
          </TouchableOpacity>

          {/* Register Button */}
          <TouchableOpacity style={styles.nextButton} onPress={handleRegister}>
            <Text style={styles.nextButtonText}>Register</Text>
          </TouchableOpacity>

          {/* Already Member */}
          <View style={styles.loginContainer}>
            <Text style={styles.alreadyMemberText}>Already a member?</Text>
            <TouchableOpacity onPress={() => router.push('/login')}>
              <Text style={styles.loginNowText}>Login now</Text>
            </TouchableOpacity>
          </View>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#77CDE0' },

  backButton: { flexDirection: 'row', alignItems: 'center', padding: 20 },
  backButtonText: { marginLeft: 5, fontSize: 16, color: '#333' },

  content: { 
    flex: 1, 
    alignItems: 'center', 
    justifyContent: 'center', 
    paddingHorizontal: 20 
  },

  brandRow: { flexDirection: 'row', alignItems: 'center', marginBottom: 10 },
  smartText: { fontSize: 40, fontWeight: 'bold', color: '#fff', marginRight: 5 },
  sightText: { fontSize: 40, fontWeight: 'bold', color: '#FFD54F', marginRight: 10 }, // ✅ yellow like login
  logo: { width: 80, height: 80, resizeMode: 'contain' },

  subBrand: { fontSize: 14, color: '#ddd', marginBottom: 20 },

  // ✅ copied from login.js
  inputGroup: {
    width: '100%',
    maxWidth: 320,
    backgroundColor: 'rgba(255,255,255,0.45)', // same as login inputContainer
    borderRadius: 20,
    padding: 25,
    marginBottom: 20,
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#C8EAF7', // same as login inputWrapper
    borderRadius: 12,
    paddingHorizontal: 15,
    marginBottom: 15,
    height: 50,
  },
  input: { flex: 1, fontSize: 16, color: '#000' },
  icon: { marginLeft: 10 },

  showPasswordText: {
    alignSelf: 'flex-end',
    width: '100%',
    maxWidth: 320,
    textAlign: 'right',
    color: '#5B4FE9',
    fontSize: 13,
    marginBottom: 20,
  },

  nextButton: {
    backgroundColor: '#FFD54F',
    paddingVertical: 15,
    paddingHorizontal: 60,
    borderRadius: 30,
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 10,
    elevation: 3,
  },
  nextButtonText: { color: '#fff', fontSize: 18, fontWeight: 'bold' },

  loginContainer: { flexDirection: 'row', marginTop: 30 },
  alreadyMemberText: { fontSize: 15, color: '#333' },
  loginNowText: { fontSize: 15, color: '#5B4FE9', fontWeight: '600', marginLeft: 5 },
});
