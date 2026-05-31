"""
LLM function tools — end_call + transfer_to_human.
Inbound-only: the caller is the first SIP participant already in the room.
"""
import asyncio
import logging
import os
from typing import Annotated, Optional

from livekit import agents, api
from livekit.agents import llm

logger = logging.getLogger("call-agent")


def create_tools(ctx: agents.JobContext):
    sip_domain = os.getenv("SIP_TRANSFER_DOMAIN")

    @llm.function_tool(
        description="Transfer the call to a human agent or specific phone number."
    )
    async def transfer_to_human(
        destination: Annotated[
            Optional[str],
            "Phone number (e.g. +91...) or SIP URI. Omit to use the default transfer number.",
        ] = None,
    ) -> str:
        if destination is None:
            destination = os.getenv("DEFAULT_TRANSFER_NUMBER")
            if not destination:
                return "No transfer number configured."

        if "@" not in destination:
            if sip_domain:
                clean = destination.replace("tel:", "").replace("sip:", "")
                destination = f"sip:{clean}@{sip_domain}"
            elif not destination.startswith(("tel:", "sip:")):
                destination = f"tel:{destination}"
        elif not destination.startswith("sip:"):
            destination = f"sip:{destination}"

        # Inbound: find the SIP caller (first sip_* participant in the room).
        participant_identity = None
        for p in ctx.room.remote_participants.values():
            if p.identity.startswith("sip_"):
                participant_identity = p.identity
                break

        if not participant_identity:
            return "Could not identify the caller for transfer."

        try:
            logger.info(f"Transferring {participant_identity} → {destination}")
            await ctx.api.sip.transfer_sip_participant(
                api.TransferSIPParticipantRequest(
                    room_name=ctx.room.name,
                    participant_identity=participant_identity,
                    transfer_to=destination,
                    play_dialtone=False,
                )
            )
            return "Transferring you now. Please hold."
        except Exception as e:
            logger.error(f"Transfer failed: {e}")
            return "Transfer failed."

    @llm.function_tool(
        description=(
            "Hang up the call. You MUST call this function to end the call — saying "
            "goodbye verbally does NOT hang up. Call this every time you say farewell."
        )
    )
    async def end_call() -> str:
        logger.info("Agent requested call end — deleting room in 4s.")

        async def _delayed_hangup():
            await asyncio.sleep(4)
            try:
                await ctx.api.room.delete_room(
                    api.DeleteRoomRequest(room=ctx.room.name)
                )
            except Exception as e:
                logger.warning(f"Could not delete room on end_call: {e}")

        asyncio.create_task(_delayed_hangup())
        return "Thank you for calling. Goodbye!"

    return [transfer_to_human, end_call]
