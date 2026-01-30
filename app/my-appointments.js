import React, { useState, useEffect, useCallback } from 'react';
import * as SecureStore from 'expo-secure-store';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  FlatList,
  Image,
  ActivityIndicator,
  RefreshControl,
  Alert
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { router } from 'expo-router';

export default function MyAppointmentsScreen() {
  const [appointments, setAppointments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [filter, setFilter] = useState("upcoming");

  useEffect(() => {
    fetchAppointments();
  }, []);

  const getAuthToken = async () => await SecureStore.getItemAsync('userToken');

  const fetchAppointments = async () => {
    setLoading(true);
    try {
      const token = await getAuthToken();
      if (!token) {
        Alert.alert("Error", "User not logged in");
        setLoading(false);
        return;
      }

      const API_URL = "https://2b7bf55b1e09.ngrok-free.app/api/my-appointments/";
      const response = await fetch(API_URL, {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Token ${token}`
        },
      });

      if (response.status === 403) {
        Alert.alert("Forbidden", "Please login again.");
        setAppointments([]);
        return;
      }

      const data = await response.json();
      setAppointments(data);
    } catch (error) {
      console.error("Error fetching appointments:", error);
      setAppointments([]);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const onRefresh = useCallback(() => {
    setRefreshing(true);
    fetchAppointments();
  }, []);

  const cancelAppointment = async (id) => {
    Alert.alert("Confirm", "Do you really want to cancel this appointment?", [
      { text: "No" },
      {
        text: "Yes",
        onPress: async () => {
          try {
            const token = await getAuthToken();
            const API_URL = `https://2b7bf55b1e09.ngrok-free.app/api/cancel-appointment/${id}/`;
            const res = await fetch(API_URL, {
              method: "PATCH",
              headers: {
                "Content-Type": "application/json",
                "Authorization": `Token ${token}`
              },
              body: JSON.stringify({ status: "Cancelled" })
            });

            if (res.ok) {
              Alert.alert("Success", "Appointment cancelled.");
              fetchAppointments();
            } else {
              Alert.alert("Error", "Failed to cancel appointment.");
            }
          } catch (err) {
            console.error(err);
            Alert.alert("Error", "Something went wrong.");
          }
        }
      }
    ]);
  };

  const deleteAppointment = async (id) => {
    Alert.alert("Confirm", "Delete this appointment permanently?", [
      { text: "No" },
      {
        text: "Yes",
        onPress: async () => {
          try {
            const token = await getAuthToken();
            const API_URL = `https://2b7bf55b1e09.ngrok-free.app/api/delete-appointment/${id}/`;
            const res = await fetch(API_URL, {
              method: "DELETE",
              headers: {
                "Content-Type": "application/json",
                "Authorization": `Token ${token}`
              }
            });

            if (res.ok) {
              Alert.alert("Deleted", "Appointment removed successfully.");
              fetchAppointments();
            } else {
              Alert.alert("Error", "Failed to delete appointment.");
            }
          } catch (err) {
            console.error(err);
            Alert.alert("Error", "Something went wrong.");
          }
        }
      }
    ]);
  };

  const editAppointment = (appointment) => {
    router.push({
      pathname: "/edit-appointment",
      params: { appointmentId: appointment.id }
    });
  };

  const filteredAppointments = appointments.filter(item => {
    const status = item.status.toLowerCase();
    const isUpcoming = ["scheduled", "confirmed", "pending"].includes(status);
    const isPast = ["completed"].includes(status);
    const isCancelled = ["cancelled"].includes(status);

    if (filter === "upcoming") return isUpcoming;
    if (filter === "past") return isPast;
    if (filter === "cancelled") return isCancelled;
    return false;
  });

  const renderAppointmentCard = (item) => (
    <View style={styles.card}>
      <Image
        source={{
          uri: `https://ui-avatars.com/api/?name=${item.doctor_name}&background=2260FF&color=fff&size=80&bold=true`,
        }}
        style={styles.avatar}
      />
      <View style={styles.info}>
        <Text style={styles.name}>{item.doctor_name}</Text>
        <Text style={styles.specialty}>Optometrist</Text>

        <Text style={styles.detailText}>👤 {item.firstName} {item.lastName}</Text>
        <Text style={styles.detailText}>🎂 {item.age} yrs | {item.gender}</Text>
        <Text style={styles.detailText}>📌 Booking For: {item.bookingFor}</Text>
        <Text style={styles.detailText}>📝 Reason: {item.reason}</Text>

        <View style={styles.row}>
          <Ionicons name="calendar-outline" size={16} color="#1E88E5" />
          <Text style={styles.detailText}>{item.date}</Text>
        </View>
        <View style={styles.row}>
          <Ionicons name="time-outline" size={16} color="#1E88E5" />
          <Text style={styles.detailText}>{item.time}</Text>
        </View>

        <View
          style={[
            styles.statusChip,
            item.status === "cancelled"
              ? { backgroundColor: "#FFCDD2" }
              : item.status === "completed"
              ? { backgroundColor: "#C8E6C9" }
              : item.status === "confirmed" || item.status === "scheduled"
              ? { backgroundColor: "#FFF9C4" }
              : { backgroundColor: "#E1BEE7" },
          ]}
        >
          <Text style={styles.statusText}>
            {item.status.toUpperCase().replace("SCHEDULED", "PENDING")}
          </Text>
        </View>

        <View style={styles.actions}>
          {(item.status === "scheduled" ||
            item.status === "pending" ||
            item.status === "confirmed") && (
            <TouchableOpacity
              style={styles.btnCancel}
              onPress={() => cancelAppointment(item.id)}
            >
              <Text style={styles.btnText}>Cancel</Text>
            </TouchableOpacity>
          )}
          {(item.status === "scheduled" || item.status === "pending") && (
            <TouchableOpacity
              style={styles.btnEdit}
              onPress={() => editAppointment(item)}
            >
              <Text style={styles.btnText}>Reschedule</Text>
            </TouchableOpacity>
          )}
          {(item.status === "cancelled" || item.status === "completed") && (
            <TouchableOpacity
              style={styles.btnDelete}
              onPress={() => deleteAppointment(item.id)}
            >
              <Text style={styles.btnText}>Delete</Text>
            </TouchableOpacity>
          )}
        </View>
      </View>
    </View>
  );

  if (loading) {
    return (
      <View style={styles.loader}>
        <ActivityIndicator size="large" color="#2260FF" />
        <Text style={{ marginTop: 10, color: "#2260FF", fontWeight: "600" }}>
          Loading appointments...
        </Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()}>
          <Ionicons name="arrow-back" size={24} color="#2260FF" />
        </TouchableOpacity>
        <Text style={styles.headerText}>My Appointments</Text>
      </View>

      <View style={styles.tabs}>
        {["upcoming", "past", "cancelled"].map(t => (
          <TouchableOpacity
            key={t}
            style={[styles.tab, filter === t && styles.activeTab]}
            onPress={() => setFilter(t)}
          >
            <Text style={[styles.tabText, filter === t && styles.activeTabText]}>
              {t.charAt(0).toUpperCase() + t.slice(1)}
            </Text>
          </TouchableOpacity>
        ))}
      </View>

      <FlatList
        data={filteredAppointments}
        keyExtractor={(item) => item.id.toString()}
        renderItem={({ item }) => renderAppointmentCard(item)}
        ListEmptyComponent={
          <Text style={styles.emptyText}>No {filter} appointments found</Text>
        }
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }
        contentContainerStyle={{ paddingBottom: 40 }}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#77CDE0' },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    backgroundColor: "#fff",
    borderBottomWidth: 1,
    borderBottomColor: "#ddd"
  },
  headerText: { fontSize: 20, fontWeight: 'bold', marginLeft: 16, color: '#2260FF' },
  loader: { flex: 1, justifyContent: "center", alignItems: "center", backgroundColor: "#77CDE0" },
  card: {
    flexDirection: 'row',
    backgroundColor: '#F5F9FF',
    margin: 12,
    borderRadius: 16,
    padding: 14,
    alignItems: 'flex-start',
    shadowColor: '#000',
    shadowOpacity: 0.1,
    shadowOffset: { width: 0, height: 3 },
    shadowRadius: 6,
    elevation: 4,
  },
  avatar: { width: 70, height: 70, borderRadius: 35, marginRight: 14 },
  info: { flex: 1 },
  name: { fontSize: 18, fontWeight: 'bold', color: '#2260FF' },
  specialty: { fontSize: 14, color: '#666', marginBottom: 6 },
  row: { flexDirection: 'row', alignItems: 'center', marginTop: 4 },
  detailText: { marginTop: 2, marginLeft: 4, fontSize: 13, color: '#333' },
  statusChip: {
    marginTop: 8,
    paddingVertical: 4,
    paddingHorizontal: 12,
    borderRadius: 20,
    alignSelf: "flex-start"
  },
  statusText: { fontWeight: "bold", fontSize: 12, color: "#222" },
  actions: { flexDirection: 'row', marginTop: 10, gap: 12 },
  btnCancel: { backgroundColor: '#E53935', paddingHorizontal: 12, paddingVertical: 6, borderRadius: 8 },
  btnEdit: { backgroundColor: '#1E88E5', paddingHorizontal: 12, paddingVertical: 6, borderRadius: 8 },
  btnDelete: { backgroundColor: '#757575', paddingHorizontal: 12, paddingVertical: 6, borderRadius: 8 },
  btnText: { color: '#fff', fontSize: 13, fontWeight: "600" },
  emptyText: { textAlign: 'center', marginTop: 40, color: '#fff', fontSize: 16, fontWeight: '600' },
  tabs: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    backgroundColor: '#C8EAF7',
    paddingVertical: 8,
    borderRadius: 20,
    marginHorizontal: 16,
    marginVertical: 10
  },
  tab: { paddingVertical: 6, paddingHorizontal: 16, borderRadius: 20 },
  activeTab: { backgroundColor: '#2260FF' },
  tabText: { fontSize: 14, color: '#444' },
  activeTabText: { color: '#fff', fontWeight: 'bold' }
});
