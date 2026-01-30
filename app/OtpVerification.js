// /app/otpverification.js

import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  Alert,
  ActivityIndicator,
} from 'react-native';
import { useRouter, useLocalSearchParams } from 'expo-router';
import * as SecureStore from 'expo-secure-store';
import { Ionicons } from '@expo/vector-icons';

const API_BASE_URL = "https://83bc02744841.ngrok-free.app/api";

export default function OtpVerification() {
  const router = useRouter();
  const { email } = useLocalSearchParams(); // ONLY email is passed

  const [otpCode, setOtpCode] = useState("");
  const [loading, setLoading] = useState(false);
  const [resendCountdown, setResendCountdown] = useState(60);

  // Countdown timer
  useEffect(() => {
    let timer;
    if (resendCountdown > 0) {
      timer = setInterval(() => {
        setResendCountdown((prev) => prev - 1);
      }, 1000);
    }
    return () => clearInterval(timer);
  }, [resendCountdown]);

  // MAIN: Verify OTP
  const handleOTPVerify = async () => {
    if (!otpCode || otpCode.length !== 6) {
      Alert.alert("Missing OTP", "Please enter the 6-digit verification code.");
      return;
    }

    setLoading(true);

    try {
      const payload = {
        email: email.trim(),
        code: otpCode.trim(),
      };

      const response = await fetch(`${API_BASE_URL}/login-verify-otp/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await response.json();
      console.log("OTP Verify Response:", data);

      if (!response.ok) {
        throw new Error(data.message || "Invalid OTP code.");
      }

      if (!data.token) {
        throw new Error("Missing token in response.");
      }

      // SAVE TOKEN
      await SecureStore.setItemAsync("userToken", data.token);

      // SAVE BASIC USER DATA
      await SecureStore.setItemAsync(
        "userProfile",
        JSON.stringify({
          email: data.email,
          name: data.name,
          id: data.id,
        })
      );

      Alert.alert("Success", "OTP Verified!", [
        { text: "Continue", onPress: () => router.replace("/dashboard") },
      ]);
    } catch (error) {
      Alert.alert("Verification Error", error.message);
      console.log("OTP Verify Error:", error);
    } finally {
      setLoading(false);
    }
  };

  // RESEND OTP
  const handleResendOtp = async () => {
    if (resendCountdown > 0) return;

    setLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/login-request-otp/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email }),
      });

      const data = await response.json();
      console.log("Resend OTP Response:", data);

      if (!response.ok) {
        throw new Error(data.message || "Failed to resend OTP.");
      }

      setResendCountdown(60);
      Alert.alert("OTP Sent", "A new OTP was sent to your email.");
    } catch (error) {
      Alert.alert("Error", error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={styles.container}>
      <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
        <Ionicons name="arrow-back" size={24} color="#000" />
      </TouchableOpacity>

      <Text style={styles.title}>OTP Verification</Text>

      <Text style={styles.instruction}>
        Enter the 6-digit code sent to:{" "}
        <Text style={styles.emailText}>{email}</Text>
      </Text>

      <View style={styles.inputContainer}>
        <View style={styles.inputWrapper}>
          <TextInput
            style={styles.input}
            placeholder="Enter 6-digit OTP"
            placeholderTextColor="#666"
            keyboardType="number-pad"
            maxLength={6}
            value={otpCode}
            onChangeText={setOtpCode}
            editable={!loading}
          />
          <Ionicons name="key-outline" size={20} color="#666" />
        </View>
      </View>

      {loading ? (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#FFD54F" />
          <Text style={styles.loadingText}>Verifying OTP...</Text>
        </View>
      ) : (
        <TouchableOpacity style={styles.button} onPress={handleOTPVerify}>
          <Text style={styles.buttonText}>Verify & Login</Text>
        </TouchableOpacity>
      )}

      <TouchableOpacity
        disabled={loading || resendCountdown > 0}
        onPress={handleResendOtp}
        style={styles.resendButton}
      >
        <Text
          style={[
            styles.resendText,
            resendCountdown > 0 && styles.resendDisabledText,
          ]}
        >
          {resendCountdown > 0
            ? `Resend in ${resendCountdown}s`
            : "Resend OTP"}
        </Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    backgroundColor: "#fff",
    justifyContent: "center",
  },
  backButton: {
    position: "absolute",
    top: 60,
    left: 20,
    zIndex: 10,
  },
  title: {
    fontSize: 24,
    fontWeight: "bold",
    color: "#000",
    textAlign: "center",
    marginBottom: 10,
  },
  instruction: {
    fontSize: 16,
    color: "#333",
    textAlign: "center",
    marginBottom: 40,
  },
  emailText: {
    fontWeight: "bold",
    color: "#000",
  },
  inputContainer: {
    marginBottom: 30,
  },
  inputWrapper: {
    flexDirection: "row",
    alignItems: "center",
    borderBottomWidth: 1,
    borderBottomColor: "#ccc",
    marginBottom: 20,
  },
  input: {
    flex: 1,
    height: 50,
    fontSize: 18,
    textAlign: "center",
    letterSpacing: 10,
    color: "#000",
    paddingRight: 10,
  },
  button: {
    backgroundColor: "#FFD54F",
    padding: 15,
    borderRadius: 10,
    alignItems: "center",
    marginBottom: 20,
  },
  buttonText: {
    color: "#000",
    fontSize: 18,
    fontWeight: "bold",
  },
  loadingContainer: {
    alignItems: "center",
    justifyContent: "center",
    height: 50,
    marginBottom: 20,
  },
  loadingText: {
    marginTop: 10,
    color: "#666",
  },
  resendButton: {
    alignItems: "center",
    padding: 10,
  },
  resendText: {
    color: "#FFD54F",
    fontWeight: "bold",
  },
  resendDisabledText: {
    color: "#999",
  },
});
